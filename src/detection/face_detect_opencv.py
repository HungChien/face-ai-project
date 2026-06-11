from pathlib import Path

import cv2


def main():
    input_path = Path("data/samples/face_test.jpg")
    output_image_path = Path("outputs/images/face_detect_opencv.jpg")
    output_report_path = Path("outputs/reports/face_detection_baseline_result.txt")

    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    output_report_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input image not found: {input_path}. "
            "Please place a test face image at data/samples/face_test.jpg"
        )

    image = cv2.imread(str(input_path))

    if image is None:
        raise ValueError(f"Failed to read image: {input_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    for idx, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            image,
            f"Face {idx + 1}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    cv2.imwrite(str(output_image_path), image)

    report_lines = [
        "Face Detection Baseline Result",
        "=" * 50,
        f"Input image: {input_path}",
        f"Output image: {output_image_path}",
        f"Number of detected faces: {len(faces)}",
        "",
        "Detected face bounding boxes:",
    ]

    if len(faces) == 0:
        report_lines.append("No faces detected.")
    else:
        for idx, (x, y, w, h) in enumerate(faces):
            report_lines.append(
                f"Face {idx + 1}: x={x}, y={y}, width={w}, height={h}"
            )

    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("Face detection baseline completed successfully.")
    print(f"Input image: {input_path}")
    print(f"Detected faces: {len(faces)}")
    print(f"Output image saved to: {output_image_path}")
    print(f"Report saved to: {output_report_path}")


if __name__ == "__main__":
    main()