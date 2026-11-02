from ultralytics import YOLO

MODELS = ['yolo26m.pt'] 
EPOCHS = 400
BATCH_SIZE = 16
OPTIMIZER = 'SGD'


for mod_variant in MODELS:

    DATASET_PATH = "dataset.yaml"

    model = YOLO(mod_variant)

    results = model.train(data = DATASET_PATH,
                          epochs = EPOCHS,
                          batch = BATCH_SIZE,
                          optimizer = OPTIMIZER,
                          hsv_v = 0,
                          degrees = 20,
                          auto_augment = None,
                          )

print(" >>>     Training successfully finished .......!!! ")
