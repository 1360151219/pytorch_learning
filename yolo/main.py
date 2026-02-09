from ultralytics import YOLO


def main():
    model = YOLO("yolo26n.pt")
    results = model.predict(
        source="https://ultralytics.com/images/bus.jpg",
        save=True,
        save_dir="images",
    )
    # for result in results:
    # result.show()


if __name__ == "__main__":
    main()
