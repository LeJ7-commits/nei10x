// WGSL for the fluid solver and the ink-wash renderer.
//
// Conventions shared by every pass:
// - uv is [0,1]^2 with y pointing down (matches pointer and framebuffer coords).
// - Velocity lives on the simulation grid, in grid cells per second.
// - Ink ("dye") lives on a finer grid as rg32float:
//     r = ink density, g = density * life. Storing life premultiplied keeps it
//     mass-weighted when advection mixes old and fresh ink.
//   rg32float isn't filterable everywhere, so dye is sampled with a manual
//   bilinear lookup instead of a sampler.

const common = /* wgsl */ `
struct Params {
  simSize: vec2f,
  dyeSize: vec2f,
  dt: f32,
  velDecay: f32,
  lifeStep: f32,
  vorticity: f32,
  aspect: f32,
  forceCount: u32,
  inkCount: u32,
  feather: f32,
};

struct Splat {
  a: vec4f, // pos.xy, brush velocity.xy (cells/s)
  b: vec4f, // ink amount, radius, drag strength, unused
};

fn bilinear(t: texture_2d<f32>, uv: vec2f, size: vec2f) -> vec4f {
  let st = uv * size - 0.5;
  let i = floor(st);
  let f = st - i;
  let hi = vec2i(size) - 1;
  let p0 = clamp(vec2i(i), vec2i(0), hi);
  let p1 = clamp(vec2i(i) + 1, vec2i(0), hi);
  let a = textureLoad(t, vec2i(p0.x, p0.y), 0);
  let b = textureLoad(t, vec2i(p1.x, p0.y), 0);
  let c = textureLoad(t, vec2i(p0.x, p1.y), 0);
  let d = textureLoad(t, vec2i(p1.x, p1.y), 0);
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
`;

const computeHeader = /* wgsl */ `
${common}
@group(0) @binding(0) var<uniform> P: Params;
`;

export const splatVelocityWGSL = /* wgsl */ `
${computeHeader}
@group(0) @binding(1) var<storage, read> splats: array<Splat>;
@group(0) @binding(2) var src: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (any(vec2f(id.xy) >= P.simSize)) { return; }
  let uv = (vec2f(id.xy) + 0.5) / P.simSize;
  var v = textureLoad(src, id.xy, 0);
  for (var i = 0u; i < P.forceCount; i++) {
    let s = splats[i];
    let d = (uv - s.a.xy) * vec2f(P.aspect, 1.0);
    let w = exp(-dot(d, d) / (s.b.y * s.b.y));
    // The brush drags nearby water toward its own velocity rather than adding
    // to it, so overlapping splats along a stroke don't pile up.
    v = vec4f(mix(v.xy, s.a.zw, w * s.b.z), 0.0, 1.0);
  }
  textureStore(dst, id.xy, v);
}
`;

export const splatInkWGSL = /* wgsl */ `
${computeHeader}
@group(0) @binding(1) var<storage, read> splats: array<Splat>;
@group(0) @binding(2) var src: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<rg32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (any(vec2f(id.xy) >= P.dyeSize)) { return; }
  let uv = (vec2f(id.xy) + 0.5) / P.dyeSize;
  var ink = textureLoad(src, id.xy, 0).rg;
  for (var i = 0u; i < P.inkCount; i++) {
    let s = splats[P.forceCount + i];
    let d = (uv - s.a.xy) * vec2f(P.aspect, 1.0);
    let w = s.b.x * exp(-dot(d, d) / (s.b.y * s.b.y));
    // Fresh ink arrives with full life.
    ink += vec2f(w, w);
  }
  // Cap how much ink can pool in one spot; life stays <= 1.
  let r = min(ink.r, 4.0);
  let g = min(ink.g * (r / max(ink.r, 1e-6)), r);
  textureStore(dst, id.xy, vec4f(r, g, 0.0, 0.0));
}
`;

export const curlWGSL = /* wgsl */ `
${computeHeader}
@group(0) @binding(1) var vel: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (any(vec2f(id.xy) >= P.simSize)) { return; }
  let p = vec2i(id.xy);
  let hi = vec2i(P.simSize) - 1;
  let L = textureLoad(vel, clamp(p - vec2i(1, 0), vec2i(0), hi), 0).y;
  let R = textureLoad(vel, clamp(p + vec2i(1, 0), vec2i(0), hi), 0).y;
  let B = textureLoad(vel, clamp(p - vec2i(0, 1), vec2i(0), hi), 0).x;
  let T = textureLoad(vel, clamp(p + vec2i(0, 1), vec2i(0), hi), 0).x;
  textureStore(dst, p, vec4f(0.5 * ((R - L) - (T - B)), 0.0, 0.0, 0.0));
}
`;

export const vorticityWGSL = /* wgsl */ `
${computeHeader}
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var curl: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (any(vec2f(id.xy) >= P.simSize)) { return; }
  let p = vec2i(id.xy);
  let hi = vec2i(P.simSize) - 1;
  let L = abs(textureLoad(curl, clamp(p - vec2i(1, 0), vec2i(0), hi), 0).x);
  let R = abs(textureLoad(curl, clamp(p + vec2i(1, 0), vec2i(0), hi), 0).x);
  let B = abs(textureLoad(curl, clamp(p - vec2i(0, 1), vec2i(0), hi), 0).x);
  let T = abs(textureLoad(curl, clamp(p + vec2i(0, 1), vec2i(0), hi), 0).x);
  let C = textureLoad(curl, p, 0).x;
  var n = 0.5 * vec2f(R - L, T - B);
  n = n / (length(n) + 1e-5);
  let force = P.vorticity * C * vec2f(n.y, -n.x);
  let v = textureLoad(src, p, 0).xy + force * P.dt;
  textureStore(dst, p, vec4f(clamp(v, vec2f(-2000.0), vec2f(2000.0)), 0.0, 1.0));
}
`;

export const divergenceWGSL = /* wgsl */ `
${computeHeader}
@group(0) @binding(1) var vel: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (any(vec2f(id.xy) >= P.simSize)) { return; }
  let p = vec2i(id.xy);
  let hi = vec2i(P.simSize) - 1;
  let C = textureLoad(vel, p, 0).xy;
  // Solid walls: mirror the normal component at the edges.
  var L = textureLoad(vel, clamp(p - vec2i(1, 0), vec2i(0), hi), 0).x;
  var R = textureLoad(vel, clamp(p + vec2i(1, 0), vec2i(0), hi), 0).x;
  var B = textureLoad(vel, clamp(p - vec2i(0, 1), vec2i(0), hi), 0).y;
  var T = textureLoad(vel, clamp(p + vec2i(0, 1), vec2i(0), hi), 0).y;
  if (p.x == 0) { L = -C.x; }
  if (p.x == hi.x) { R = -C.x; }
  if (p.y == 0) { B = -C.y; }
  if (p.y == hi.y) { T = -C.y; }
  textureStore(dst, p, vec4f(0.5 * (R - L + T - B), 0.0, 0.0, 0.0));
}
`;

export const jacobiWGSL = /* wgsl */ `
${computeHeader}
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var divergence: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (any(vec2f(id.xy) >= P.simSize)) { return; }
  let p = vec2i(id.xy);
  let hi = vec2i(P.simSize) - 1;
  let L = textureLoad(src, clamp(p - vec2i(1, 0), vec2i(0), hi), 0).x;
  let R = textureLoad(src, clamp(p + vec2i(1, 0), vec2i(0), hi), 0).x;
  let B = textureLoad(src, clamp(p - vec2i(0, 1), vec2i(0), hi), 0).x;
  let T = textureLoad(src, clamp(p + vec2i(0, 1), vec2i(0), hi), 0).x;
  let d = textureLoad(divergence, p, 0).x;
  textureStore(dst, p, vec4f((L + R + B + T - d) * 0.25, 0.0, 0.0, 0.0));
}
`;

export const gradientWGSL = /* wgsl */ `
${computeHeader}
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var pressure: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (any(vec2f(id.xy) >= P.simSize)) { return; }
  let p = vec2i(id.xy);
  let hi = vec2i(P.simSize) - 1;
  let L = textureLoad(pressure, clamp(p - vec2i(1, 0), vec2i(0), hi), 0).x;
  let R = textureLoad(pressure, clamp(p + vec2i(1, 0), vec2i(0), hi), 0).x;
  let B = textureLoad(pressure, clamp(p - vec2i(0, 1), vec2i(0), hi), 0).x;
  let T = textureLoad(pressure, clamp(p + vec2i(0, 1), vec2i(0), hi), 0).x;
  var v = textureLoad(src, p, 0).xy - 0.5 * vec2f(R - L, T - B);
  if (p.x == 0 || p.x == hi.x) { v.x = 0.0; }
  if (p.y == 0 || p.y == hi.y) { v.y = 0.0; }
  textureStore(dst, p, vec4f(v, 0.0, 1.0));
}
`;

export const advectVelocityWGSL = /* wgsl */ `
${computeHeader}
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (any(vec2f(id.xy) >= P.simSize)) { return; }
  let uv = (vec2f(id.xy) + 0.5) / P.simSize;
  let v = textureLoad(src, id.xy, 0).xy;
  let back = uv - P.dt * v / P.simSize;
  let res = textureSampleLevel(src, samp, back, 0.0).xy * P.velDecay;
  textureStore(dst, id.xy, vec4f(res, 0.0, 1.0));
}
`;

// MacCormack ink advection, step 1: plain semi-Lagrangian prediction.
export const advectInkPredictWGSL = /* wgsl */ `
${computeHeader}
@group(0) @binding(1) var vel: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var ink: texture_2d<f32>;
@group(0) @binding(4) var dst: texture_storage_2d<rg32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (any(vec2f(id.xy) >= P.dyeSize)) { return; }
  let uv = (vec2f(id.xy) + 0.5) / P.dyeSize;
  let v = textureSampleLevel(vel, samp, uv, 0.0).xy;
  let back = uv - P.dt * v / P.simSize;
  textureStore(dst, id.xy, bilinear(ink, back, P.dyeSize));
}
`;

// MacCormack step 2: error-correct, clamp to stay monotone, then feather the
// ink (slow bleeding, as on wet xuan paper) and age it.
export const advectInkCorrectWGSL = /* wgsl */ `
${computeHeader}
@group(0) @binding(1) var vel: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var ink: texture_2d<f32>;
@group(0) @binding(4) var predicted: texture_2d<f32>;
@group(0) @binding(5) var dst: texture_storage_2d<rg32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (any(vec2f(id.xy) >= P.dyeSize)) { return; }
  let p = vec2i(id.xy);
  let hi = vec2i(P.dyeSize) - 1;
  let uv = (vec2f(id.xy) + 0.5) / P.dyeSize;
  let disp = P.dt * textureSampleLevel(vel, samp, uv, 0.0).xy / P.simSize;

  // Only density gets the MacCormack correction. Life rides along as a ratio
  // from the smooth prediction: correcting it separately lets it hit zero at
  // ink edges and would erase ink that isn't old yet.
  let hat = textureLoad(predicted, p, 0).rg;
  let bar = bilinear(predicted, uv + disp, P.dyeSize).r;
  let orig = textureLoad(ink, p, 0).r;

  // Clamp to the four source texels the prediction interpolated between.
  let st = (uv - disp) * P.dyeSize - 0.5;
  let i0 = clamp(vec2i(floor(st)), vec2i(0), hi);
  let i1 = clamp(vec2i(floor(st)) + 1, vec2i(0), hi);
  let a = textureLoad(ink, vec2i(i0.x, i0.y), 0).r;
  let b = textureLoad(ink, vec2i(i1.x, i0.y), 0).r;
  let c = textureLoad(ink, vec2i(i0.x, i1.y), 0).r;
  let d = textureLoad(ink, vec2i(i1.x, i1.y), 0).r;
  var r = clamp(hat.r + 0.5 * (orig - bar), min(min(a, b), min(c, d)), max(max(a, b), max(c, d)));
  let lifeHat = select(1.0, clamp(hat.g / hat.r, 0.0, 1.0), hat.r > 1e-6);
  var res = vec2f(r, r * lifeHat);

  // Feathering: a little diffusion toward the neighbourhood average.
  let nL = textureLoad(predicted, clamp(p - vec2i(1, 0), vec2i(0), hi), 0).rg;
  let nR = textureLoad(predicted, clamp(p + vec2i(1, 0), vec2i(0), hi), 0).rg;
  let nB = textureLoad(predicted, clamp(p - vec2i(0, 1), vec2i(0), hi), 0).rg;
  let nT = textureLoad(predicted, clamp(p + vec2i(0, 1), vec2i(0), hi), 0).rg;
  res = mix(res, 0.25 * (nL + nR + nB + nT), P.feather);

  r = max(res.r, 0.0);
  var life = select(0.0, clamp(res.g / r, 0.0, 1.0), r > 1e-6);
  if (P.lifeStep > 0.0) {
    life -= P.lifeStep;
    if (life <= 0.0) { r = 0.0; }
  }
  if (r < 1e-4) { r = 0.0; }
  let g = r * max(life, 0.0);
  textureStore(dst, p, vec4f(r, g, 0.0, 0.0));
}
`;

export const renderWGSL = /* wgsl */ `
${common}
struct View {
  size: vec2f,
  time: f32,
  hold: f32, // 1 = ink never fades, otherwise fraction of life spent fading
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<uniform> V: View;
@group(0) @binding(2) var ink: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

fn hash(p: vec2f) -> f32 {
  return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453);
}

fn noise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2f(1.0, 0.0)), u.x),
             mix(hash(i + vec2f(0.0, 1.0)), hash(i + vec2f(1.0, 1.0)), u.x), u.y);
}

fn fbm(p: vec2f) -> f32 {
  var v = 0.0;
  var a = 0.5;
  var q = p;
  for (var i = 0; i < 4; i++) {
    v += a * noise(q);
    q = q * 2.03 + vec2f(1.7, 9.2);
    a *= 0.5;
  }
  return v;
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let uv = pos.xy / V.size;
  let q = (uv - 0.5) * vec2f(P.aspect, 1.0);

  // Water: dim slate lit softly from below, with drifting pockets of deep green.
  let glow = exp(-dot(q, q) * 2.2);
  var water = mix(vec3f(0.050, 0.068, 0.076), vec3f(0.250, 0.305, 0.320), glow);
  let t = V.time * 0.02;
  let green = smoothstep(0.45, 0.8, fbm(q * 1.6 + vec2f(t, -t * 0.7)));
  water = mix(water, vec3f(0.070, 0.185, 0.130), green * 0.6 * (0.4 + 0.6 * glow));
  water *= 0.94 + 0.06 * fbm(q * 5.0 - vec2f(t * 3.0, t * 2.0));

  // Ink: fully visible while young, fading over the last part of its life.
  let s = bilinear(ink, uv, P.dyeSize).rg;
  let life = s.g / max(s.r, 1e-5);
  let vis = select(s.r * smoothstep(0.0, V.hold, life), s.r, V.hold >= 1.0);
  // Ink-wash tone curve: pale washes stay readable, heavy ink goes to black.
  let density = 1.0 - exp(-1.6 * pow(vis, 0.85));
  // Thin, watery ink carries a cool green cast; heavy ink is true black.
  let inkColor = mix(vec3f(0.020, 0.060, 0.045), vec3f(0.004, 0.006, 0.006),
                     smoothstep(0.15, 0.7, density));
  var col = mix(water, inkColor, density);

  let vig = 1.0 - 0.35 * smoothstep(0.35, 1.0, length(q));
  col *= vig;
  col += (hash(pos.xy + fract(V.time)) - 0.5) / 255.0;
  return vec4f(col, 1.0);
}
`;
