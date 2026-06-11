from pathlib import Path

import cv2


def main():
    input_path = Path("data/samples/test.jpg")
    output_dir = Path("outputs/images")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input image not found: {input_path}. "
            "Please place a test image at data/samples/test.jpg"
        )

    image = cv2.imread(str(input_path))

    if image is None:
        raise ValueError(f"Failed to read image: {input_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 50, 150)

    gray_path = output_dir / "gray.jpg"
    blur_path = output_dir / "blur.jpg"
    edges_path = output_dir / "edges.jpg"

    cv2.imwrite(str(gray_path), gray)
    cv2.imwrite(str(blur_path), blur)
    cv2.imwrite(str(edges_path), edges)

    print("OpenCV demo completed successfully.")
    print(f"Input image: {input_path}")
    print(f"Gray image saved to: {gray_path}")
    print(f"Blur image saved to: {blur_path}")
    print(f"Edges image saved to: {edges_path}")


if __name__ == "__main__":
    main()