import os
from PIL import Image
import numpy as np

UNICODE_OFFSET = 1072

WIDTH = 12
HEIGHT = 14

MIN = 0
MAX = 210


class PreparationData:
    def __init__(self, dir_path):
        self.dir_path = dir_path

    def rename_files(self, file_new_name):
        file_paths = [file for file in os.listdir(self.dir_path) if os.path.isfile(os.path.join(self.dir_path, file))
                      and os.path.splitext(file)[1] == '.jpg']

        for i in range(len(file_paths)):
            file_paths[i] = self.dir_path + '/' + file_paths[i]
            file_name = file_paths[i].split()[0]
            file_new_path = os.path.join(file_name + '_' + str(i) + '.jpg')
            os.rename(file_paths[i], file_new_path)

        print("rename_files: DONE")

    def create_dataset(self):
        file_paths = [file for file in os.listdir(self.dir_path) if os.path.isfile(os.path.join(self.dir_path, file))
                      and os.path.splitext(file)[1] == '.jpg']
        file_paths.sort()

        # Create files with changed pixels
        i = 0
        for file_path in file_paths:
            for x in range(WIDTH):
                image_1 = Image.open(self.dir_path + '/' + file_path)

                for y in range(HEIGHT):
                    if image_1.getpixel((x, y)) < (140, 140, 140):
                        image_1.putpixel((x, y), (200, 200, 200))

                    image_2 = Image.open(self.dir_path + '/' + file_path)
                    if image_2.getpixel((x, y)) < (140, 140, 140):
                        image_2.putpixel((x, y), (200, 200, 200))
                        image_2.save(self.dir_path + '/' + os.path.splitext(file_path)[0] +
                                     '_' + str(i) + '.jpg')
                        i += 1

                    image_3 = Image.open(self.dir_path + '/' + file_path)
                    if image_3.getpixel((x, y)) < (140, 140, 140):
                        image_3.putpixel((x, y), (200, 200, 200))
                        image_3.putpixel((x - 1, y), (200, 200, 200))
                        image_3.putpixel((x, y - 1), (200, 200, 200))
                        image_3.putpixel((x - 1, y - 1), (200, 200, 200))
                        image_3.putpixel((x + 1, y), (200, 200, 200))
                        image_3.putpixel((x, y + 1), (200, 200, 200))
                        image_3.putpixel((x + 1, y + 1), (200, 200, 200))
                        image_3.save(self.dir_path + '/' + os.path.splitext(file_path)[0] +
                                     '_' + str(i) + '.jpg')
                        i += 1

                image_1.save(self.dir_path + '/' + os.path.splitext(file_path)[0] +
                             '_' + str(i) + '.jpg')
                i += 1

        print("prepare_data: DONE")

    def create_matrix(self):
        file_paths = [file for file in os.listdir(self.dir_path) if os.path.isfile(os.path.join(self.dir_path, file))
                      and os.path.splitext(file)[1] == '.jpg']
        file_paths.sort()
        matrix_result = []
        for file_path in file_paths:
            image = Image.open(self.dir_path + '/' + file_path)
            vector_image = list()
            vector_image.append(ord(file_path[0]) - UNICODE_OFFSET)
            for y in range(HEIGHT):
                for x in range(WIDTH):
                    image_rgb = image.getpixel((x, y))
                    median_image_rgb = 0
                    for color in image_rgb:
                        median_image_rgb += color
                    median_image_rgb /= 3
                    vector_image.append(median_image_rgb / (MAX - MIN))
            matrix_result.append(vector_image)
        matrix_result = np.array(matrix_result)
        return matrix_result
