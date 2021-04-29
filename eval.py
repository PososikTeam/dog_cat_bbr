import torch
from model import get_model
from augmentation import get_test_transform
import pandas as pd
import time
import cv2




def main():
    image_size = (512, 512)
    test_df = pd.read_csv('test.csv')


    model = get_model('mobilenetv2_120d')
    model.load_state_dict(torch.load('runs//Apr28_19_19_mobilenetv2_120d_512_medium_adam_0.0003_EfficientIoU//checkpoints//best_full.pth')['model_state_dict'])
    model.eval()

    num = 10
    img = cv2.cvtColor(cv2.imread('data/'+ test_df['name'][num]), cv2.COLOR_BGR2RGB)
    bbox = [test_df['xmin'][num], test_df['ymin'][num], test_df['xmax'][num], test_df['ymax'][num]]
    label = test_df['id'][num]

    test_transform = get_test_transform(image_size)

    torch_img = test_transform(image = img, bboxes=[bbox], category_ids=[label])['image']
    torch_img = torch.unsqueeze(torch_img, 0)

    start_time = time.time()
    with torch.no_grad():
        pred = model(torch_img)

    print('Time = ', time.time() - start_time)


if __name__ == '__main__':
    main()