from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(
    data='coco8.yaml',
    epochs=50,
    freeze=10,
    plots=True
)
sample_image = 'https://ultralytics.com/images/bus.jpg'
prediction = model.predict(source=sample_image, show=True, save=True)