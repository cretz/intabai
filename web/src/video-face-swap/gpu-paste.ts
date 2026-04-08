// WebGPU compute shader for paste-back. Replaces the JS double-loop in
// cv.ts:pasteBack which is the single biggest CPU bottleneck per frame on
// mobile (operates on the full output frame, ~2M pixels at 1080p).
//
// The shader does: for every output pixel, map back through the affine
// matrix to crop space, bilinear-sample the crop and the mask, blend with
// the original frame pixel, write to the output buffer.
//
// One device + pipeline are created lazily on first use and reused for the
// whole session. Per-frame allocations are cached by frame/crop dimensions.

const WGSL = /* wgsl */ `
struct Params {
  fw: u32,
  fh: u32,
  cw: u32,
  ch: u32,
  m0: f32, m1: f32, m2: f32,
  m3: f32, m4: f32, m5: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> frameIn: array<u32>;
@group(0) @binding(2) var<storage, read> cropIn: array<u32>;
@group(0) @binding(3) var<storage, read> maskIn: array<f32>;
@group(0) @binding(4) var<storage, read_write> outImg: array<u32>;

fn unpackRGBA(p: u32) -> vec4<f32> {
  return vec4<f32>(
    f32(p & 0xffu),
    f32((p >> 8u) & 0xffu),
    f32((p >> 16u) & 0xffu),
    f32((p >> 24u) & 0xffu),
  );
}

fn packRGBA(v: vec4<f32>) -> u32 {
  let c = clamp(v, vec4<f32>(0.0), vec4<f32>(255.0));
  return u32(c.x) | (u32(c.y) << 8u) | (u32(c.z) << 16u) | (u32(c.w) << 24u);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.fw || y >= params.fh) { return; }
  let idx = y * params.fw + x;
  let framePix = unpackRGBA(frameIn[idx]);

  let fx = f32(x);
  let fy = f32(y);
  let cx = params.m0 * fx + params.m1 * fy + params.m2;
  let cy = params.m3 * fx + params.m4 * fy + params.m5;
  let cwf = f32(params.cw - 1u);
  let chf = f32(params.ch - 1u);
  if (cx < 0.0 || cx >= cwf || cy < 0.0 || cy >= chf) {
    outImg[idx] = packRGBA(framePix);
    return;
  }

  let x0 = i32(floor(cx));
  let y0 = i32(floor(cy));
  let x1 = x0 + 1;
  let y1 = y0 + 1;
  let frx = cx - f32(x0);
  let fry = cy - f32(y0);
  let cwi = i32(params.cw);

  let mi = maskIn[y0 * cwi + x0] * (1.0 - frx) * (1.0 - fry)
         + maskIn[y0 * cwi + x1] * frx * (1.0 - fry)
         + maskIn[y1 * cwi + x0] * (1.0 - frx) * fry
         + maskIn[y1 * cwi + x1] * frx * fry;
  if (mi < 0.001) {
    outImg[idx] = packRGBA(framePix);
    return;
  }

  let p00 = unpackRGBA(cropIn[y0 * cwi + x0]);
  let p10 = unpackRGBA(cropIn[y0 * cwi + x1]);
  let p01 = unpackRGBA(cropIn[y1 * cwi + x0]);
  let p11 = unpackRGBA(cropIn[y1 * cwi + x1]);
  let cropPix = p00 * (1.0 - frx) * (1.0 - fry)
              + p10 * frx * (1.0 - fry)
              + p01 * (1.0 - frx) * fry
              + p11 * frx * fry;

  let blended = framePix * (1.0 - mi) + cropPix * mi;
  outImg[idx] = packRGBA(vec4<f32>(blended.xyz, framePix.w));
}
`;

interface FrameBuffers {
  fw: number;
  fh: number;
  frameBuf: GPUBuffer;
  outBuf: GPUBuffer;
  stagingBuf: GPUBuffer;
}

interface CropBuffers {
  cw: number;
  ch: number;
  cropBuf: GPUBuffer;
  maskBuf: GPUBuffer;
}

export class GpuPasteBack {
  private readonly device: GPUDevice;
  private readonly pipeline: GPUComputePipeline;
  private readonly bindLayout: GPUBindGroupLayout;
  private readonly uniformBuf: GPUBuffer;
  private frame: FrameBuffers | null = null;
  private crop: CropBuffers | null = null;

  /** True if WebGPU is available in this context. */
  static isSupported(): boolean {
    return typeof navigator !== "undefined" && !!navigator.gpu;
  }

  /**
   * Acquire a GPUDevice. Callers should obtain one device per pipeline and
   * pass it to all GPU consumers (gpu-paste, ORT WebGPU EP, etc.) so that
   * everything shares the same device + queue.
   */
  static async requestDevice(): Promise<GPUDevice> {
    if (!navigator.gpu) throw new Error("WebGPU not supported");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("WebGPU adapter unavailable");
    return adapter.requestDevice();
  }

  constructor(device: GPUDevice) {
    this.device = device;
    const module = device.createShaderModule({ code: WGSL });
    this.bindLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [this.bindLayout] });
    this.pipeline = device.createComputePipeline({
      layout,
      compute: { module, entryPoint: "main" },
    });
    this.uniformBuf = device.createBuffer({
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  private ensureFrameBuffers(fw: number, fh: number): FrameBuffers {
    if (this.frame && this.frame.fw === fw && this.frame.fh === fh) return this.frame;
    if (this.frame) {
      this.frame.frameBuf.destroy();
      this.frame.outBuf.destroy();
      this.frame.stagingBuf.destroy();
    }
    const device = this.device;
    const byteSize = fw * fh * 4;
    this.frame = {
      fw,
      fh,
      frameBuf: device.createBuffer({
        size: byteSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }),
      outBuf: device.createBuffer({
        size: byteSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      }),
      stagingBuf: device.createBuffer({
        size: byteSize,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      }),
    };
    return this.frame;
  }

  private ensureCropBuffers(cw: number, ch: number): CropBuffers {
    if (this.crop && this.crop.cw === cw && this.crop.ch === ch) return this.crop;
    if (this.crop) {
      this.crop.cropBuf.destroy();
      this.crop.maskBuf.destroy();
    }
    const device = this.device;
    this.crop = {
      cw,
      ch,
      cropBuf: device.createBuffer({
        size: cw * ch * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }),
      maskBuf: device.createBuffer({
        size: cw * ch * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }),
    };
    return this.crop;
  }

  /**
   * GPU paste-back. Returns a new ImageData with the swapped/enhanced crop
   * blended into the original frame using the supplied mask and affine.
   * Falls back to the JS implementation if WebGPU is unavailable or errors.
   */
  async apply(
    frame: ImageData,
    crop: ImageData,
    affineMatrix: number[],
    mask: Float32Array,
  ): Promise<ImageData> {
    const device = this.device;
    const fw = frame.width;
    const fh = frame.height;
    const cw = crop.width;
    const ch = crop.height;

    const fbufs = this.ensureFrameBuffers(fw, fh);
    const cbufs = this.ensureCropBuffers(cw, ch);

    // Upload frame, crop, mask, and uniforms.
    device.queue.writeBuffer(
      fbufs.frameBuf,
      0,
      frame.data.buffer,
      frame.data.byteOffset,
      fw * fh * 4,
    );
    device.queue.writeBuffer(cbufs.cropBuf, 0, crop.data.buffer, crop.data.byteOffset, cw * ch * 4);
    device.queue.writeBuffer(cbufs.maskBuf, 0, mask.buffer, mask.byteOffset, cw * ch * 4);

    const params = new ArrayBuffer(48);
    const u32 = new Uint32Array(params);
    const f32 = new Float32Array(params);
    u32[0] = fw;
    u32[1] = fh;
    u32[2] = cw;
    u32[3] = ch;
    f32[4] = affineMatrix[0];
    f32[5] = affineMatrix[1];
    f32[6] = affineMatrix[2];
    f32[7] = affineMatrix[3];
    f32[8] = affineMatrix[4];
    f32[9] = affineMatrix[5];
    device.queue.writeBuffer(this.uniformBuf, 0, params);

    const bindGroup = device.createBindGroup({
      layout: this.bindLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuf } },
        { binding: 1, resource: { buffer: fbufs.frameBuf } },
        { binding: 2, resource: { buffer: cbufs.cropBuf } },
        { binding: 3, resource: { buffer: cbufs.maskBuf } },
        { binding: 4, resource: { buffer: fbufs.outBuf } },
      ],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(fw / 8), Math.ceil(fh / 8));
    pass.end();
    encoder.copyBufferToBuffer(fbufs.outBuf, 0, fbufs.stagingBuf, 0, fw * fh * 4);
    device.queue.submit([encoder.finish()]);

    await fbufs.stagingBuf.mapAsync(GPUMapMode.READ);
    const mapped = new Uint8ClampedArray(fbufs.stagingBuf.getMappedRange().slice(0));
    fbufs.stagingBuf.unmap();
    return new ImageData(mapped, fw, fh);
  }

  /**
   * Release per-frame/per-crop buffers and the uniform buffer. Does NOT
   * destroy the GPUDevice - the device is owned by the caller and may be
   * shared with other consumers (ORT WebGPU EP, etc.).
   */
  destroy(): void {
    if (this.frame) {
      this.frame.frameBuf.destroy();
      this.frame.outBuf.destroy();
      this.frame.stagingBuf.destroy();
      this.frame = null;
    }
    if (this.crop) {
      this.crop.cropBuf.destroy();
      this.crop.maskBuf.destroy();
      this.crop = null;
    }
    this.uniformBuf.destroy();
  }
}
