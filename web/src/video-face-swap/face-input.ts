export class FaceInput {
  private img: HTMLImageElement;
  private currentObjectUrl: string | null = null;

  constructor(
    private fileInput: HTMLInputElement,
    private previewContainer: HTMLDivElement,
  ) {
    this.img = document.createElement("img");
    this.img.style.maxWidth = "200px";
    this.img.style.display = "none";
    this.previewContainer.appendChild(this.img);

    this.fileInput.addEventListener("change", () => this.onFileChange());
  }

  private onFileChange(): void {
    const file = this.fileInput.files?.[0];
    if (!file) return;
    if (this.currentObjectUrl) URL.revokeObjectURL(this.currentObjectUrl);
    this.currentObjectUrl = URL.createObjectURL(file);
    this.img.src = this.currentObjectUrl;
    this.img.style.display = "block";
  }

  getImage(): HTMLImageElement | null {
    return this.img.src ? this.img : null;
  }

  getImageData(): ImageData | null {
    if (!this.img.src || !this.img.naturalWidth) return null;
    const canvas = document.createElement("canvas");
    canvas.width = this.img.naturalWidth;
    canvas.height = this.img.naturalHeight;
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(this.img, 0, 0);
    return ctx.getImageData(0, 0, canvas.width, canvas.height);
  }
}
