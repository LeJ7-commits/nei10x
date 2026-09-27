import * as S from './shaders';

export interface Splat {
  x: number; // uv
  y: number;
  fx: number; // force, grid cells per second
  fy: number;
  ink: number;
  radius: number; // fraction of the canvas height
  /** 0..1: how strongly water under the brush is pulled to the brush velocity. */
  drag?: number;
}

export interface StepSettings {
  dt: number;
  /** Seconds until ink has fully faded; Infinity keeps it forever. */
  inkLifetime: number;
}

const MAX_SPLATS = 256;
const WG = 8;

// Tuning. Velocity half-life is short so the water settles within a few
// seconds; ink is kept by the lifetime setting instead.
const VELOCITY_HALF_LIFE = 0.7;
const VORTICITY = 10;
const PRESSURE_ITERATIONS = 28;
const FEATHER_PER_SECOND = 1.2; // mixed toward neighbours at 60 fps ≈ 0.02/frame
/** Fraction of the ink lifetime spent fading; before that it holds fully. */
export const FADE_FRACTION = 0.35;

interface Pair {
  read: GPUTexture;
  write: GPUTexture;
  swap(): void;
  index: number;
}

function pair(device: GPUDevice, w: number, h: number, format: GPUTextureFormat, label: string): Pair {
  const make = (i: number) =>
    device.createTexture({
      label: `${label}${i}`,
      size: [w, h],
      format,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
    });
  const t = [make(0), make(1)];
  const p: Pair = {
    read: t[0],
    write: t[1],
    index: 0,
    swap() {
      p.index ^= 1;
      p.read = t[p.index];
      p.write = t[p.index ^ 1];
    },
  };
  return p;
}

export class Fluid {
  readonly simW: number;
  readonly simH: number;
  readonly dyeW: number;
  readonly dyeH: number;
  readonly aspect: number;

  private device: GPUDevice;
  private params: GPUBuffer;
  private paramData = new ArrayBuffer(48);
  private splatBuffer: GPUBuffer;
  private splatData = new Float32Array(MAX_SPLATS * 8);
  private sampler: GPUSampler;

  private vel: Pair;
  private pressure: Pair;
  private ink: Pair;
  private inkPredicted: GPUTexture;
  private curl: GPUTexture;
  private divergence: GPUTexture;

  private pipes: Record<string, GPUComputePipeline> = {};
  private bindGroups = new Map<string, GPUBindGroup>();
  private textures: GPUTexture[] = [];

  constructor(device: GPUDevice, aspect: number, simShort: number, dyeShort: number) {
    this.device = device;
    this.aspect = aspect;
    const dims = (short: number) =>
      aspect >= 1
        ? [Math.round(short * aspect), short]
        : [short, Math.round(short / aspect)];
    [this.simW, this.simH] = dims(simShort);
    [this.dyeW, this.dyeH] = dims(dyeShort);

    this.params = device.createBuffer({
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.splatBuffer = device.createBuffer({
      size: this.splatData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.sampler = device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

    this.vel = pair(device, this.simW, this.simH, 'rgba16float', 'vel');
    this.pressure = pair(device, this.simW, this.simH, 'r32float', 'pressure');
    this.ink = pair(device, this.dyeW, this.dyeH, 'rg32float', 'ink');
    const single = (format: GPUTextureFormat, w: number, h: number, label: string) =>
      device.createTexture({
        label,
        size: [w, h],
        format,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
      });
    this.inkPredicted = single('rg32float', this.dyeW, this.dyeH, 'inkPredicted');
    this.curl = single('r32float', this.simW, this.simH, 'curl');
    this.divergence = single('r32float', this.simW, this.simH, 'divergence');
    this.textures = [
      this.vel.read, this.vel.write, this.pressure.read, this.pressure.write,
      this.ink.read, this.ink.write, this.inkPredicted, this.curl, this.divergence,
    ];

    const sources: Record<string, string> = {
      splatVelocity: S.splatVelocityWGSL,
      splatInk: S.splatInkWGSL,
      curl: S.curlWGSL,
      vorticity: S.vorticityWGSL,
      divergence: S.divergenceWGSL,
      jacobi: S.jacobiWGSL,
      gradient: S.gradientWGSL,
      advectVelocity: S.advectVelocityWGSL,
      advectInkPredict: S.advectInkPredictWGSL,
      advectInkCorrect: S.advectInkCorrectWGSL,
    };
    for (const [name, code] of Object.entries(sources)) {
      this.pipes[name] = device.createComputePipeline({
        label: name,
        layout: 'auto',
        compute: { module: device.createShaderModule({ label: name, code }), entryPoint: 'main' },
      });
    }
  }

  get paramsBuffer(): GPUBuffer {
    return this.params;
  }

  get inkTexture(): GPUTexture {
    return this.ink.read;
  }

  /** Changes whenever inkTexture points at the other half of the ping-pong pair. */
  get inkIndex(): number {
    return this.ink.index;
  }

  destroy(): void {
    for (const t of this.textures) t.destroy();
    this.params.destroy();
    this.splatBuffer.destroy();
  }

  /** Writes the uniform block. Also used by the renderer, so call it even when paused. */
  writeParams(dt: number, inkLifetime: number, forceCount = 0, inkCount = 0): void {
    const f = new Float32Array(this.paramData);
    const u = new Uint32Array(this.paramData);
    f[0] = this.simW;
    f[1] = this.simH;
    f[2] = this.dyeW;
    f[3] = this.dyeH;
    f[4] = dt;
    f[5] = Math.pow(0.5, dt / VELOCITY_HALF_LIFE);
    f[6] = Number.isFinite(inkLifetime) ? dt / inkLifetime : 0;
    f[7] = VORTICITY;
    f[8] = this.aspect;
    u[9] = forceCount;
    u[10] = inkCount;
    f[11] = Math.min(FEATHER_PER_SECOND * dt, 0.2);
    this.device.queue.writeBuffer(this.params, 0, this.paramData);
  }

  step(encoder: GPUCommandEncoder, splats: Splat[], s: StepSettings): void {
    const forces = splats.filter((p) => p.fx !== 0 || p.fy !== 0).slice(0, MAX_SPLATS / 2);
    const inks = splats.filter((p) => p.ink > 0).slice(0, MAX_SPLATS - forces.length);
    [...forces, ...inks].forEach((p, i) => {
      this.splatData.set([p.x, p.y, p.fx, p.fy, p.ink, p.radius, p.drag ?? 0, 0], i * 8);
    });
    if (forces.length + inks.length > 0) {
      this.device.queue.writeBuffer(this.splatBuffer, 0, this.splatData, 0, (forces.length + inks.length) * 8);
    }
    this.writeParams(s.dt, s.inkLifetime, forces.length, inks.length);

    const pass = encoder.beginComputePass();
    const simGroups: [number, number] = [Math.ceil(this.simW / WG), Math.ceil(this.simH / WG)];
    const dyeGroups: [number, number] = [Math.ceil(this.dyeW / WG), Math.ceil(this.dyeH / WG)];
    const run = (name: string, key: string, entries: () => GPUBindingResource[], groups: [number, number]) => {
      pass.setPipeline(this.pipes[name]);
      pass.setBindGroup(0, this.group(name, key, () => [{ buffer: this.params }, ...entries()]));
      pass.dispatchWorkgroups(groups[0], groups[1]);
    };
    const v = this.vel;
    const ink = this.ink;
    const pr = this.pressure;
    const view = (t: GPUTexture) => t.createView();

    if (forces.length) {
      run('splatVelocity', `${v.index}`, () => [{ buffer: this.splatBuffer }, view(v.read), view(v.write)], simGroups);
      v.swap();
    }
    if (inks.length) {
      run('splatInk', `${ink.index}`, () => [{ buffer: this.splatBuffer }, view(ink.read), view(ink.write)], dyeGroups);
      ink.swap();
    }

    run('curl', `${v.index}`, () => [view(v.read), view(this.curl)], simGroups);
    run('vorticity', `${v.index}`, () => [view(v.read), view(this.curl), view(v.write)], simGroups);
    v.swap();

    run('divergence', `${v.index}`, () => [view(v.read), view(this.divergence)], simGroups);
    for (let i = 0; i < PRESSURE_ITERATIONS; i++) {
      run('jacobi', `${pr.index}`, () => [view(pr.read), view(this.divergence), view(pr.write)], simGroups);
      pr.swap();
    }
    run('gradient', `${v.index}:${pr.index}`, () => [view(v.read), view(pr.read), view(v.write)], simGroups);
    v.swap();

    run('advectInkPredict', `${v.index}:${ink.index}`,
      () => [view(v.read), this.sampler, view(ink.read), view(this.inkPredicted)], dyeGroups);
    run('advectInkCorrect', `${v.index}:${ink.index}`,
      () => [view(v.read), this.sampler, view(ink.read), view(this.inkPredicted), view(ink.write)], dyeGroups);
    ink.swap();

    run('advectVelocity', `${v.index}`, () => [view(v.read), this.sampler, view(v.write)], simGroups);
    v.swap();

    pass.end();
  }

  /** Bind groups are cached per pipeline and ping-pong parity. */
  private group(name: string, key: string, resources: () => GPUBindingResource[]): GPUBindGroup {
    const id = `${name}:${key}`;
    let g = this.bindGroups.get(id);
    if (!g) {
      g = this.device.createBindGroup({
        label: id,
        layout: this.pipes[name].getBindGroupLayout(0),
        entries: resources().map((resource, binding) => ({ binding, resource })),
      });
      this.bindGroups.set(id, g);
    }
    return g;
  }
}
