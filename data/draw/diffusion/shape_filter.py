import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Any
from copy import deepcopy
import os
from tqdm import tqdm
from multiprocessing import Pool, Pipe
import random
from iterwrap import iterate_wrapper
from PIL import Image, ImageColor

size_ = (128, 128)


def find_contour(img):
    if type(img) == str:
        img_rgb = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    else:
        img_rgb = np.asarray(Image.fromarray(img).resize(size_, Image.LANCZOS))
    img_gray = img_rgb[:, :, 3]
    contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # img2=np.zeros(img_rgb.shape)*(0,0,0,0)
    # cv2.drawContours(img2,[max(contours,key=cv2.contourArea)],0,(255,255,255,255),thickness=10)
    # cv2.imwrite("test.png",img2)
    if len(contours) == 0:
        return "random"
    return max(contours, key=cv2.contourArea)


def find_contour_png(png):
    try:
        img_gray = png[:, :, 3]
        contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return max(contours, key=cv2.contourArea)
    except:
        pass
    return None


def calculateSimilarity(mysc, query_contour, batched_file):
    contour = find_contour_png(pngs[batched_file])

    return batched_file, mysc.computeDistance(query_contour, contour), contour
    # return batched_file,cv2.matchShapes(query_contour,contour,1,0),contour # type: ignore


def calculateSimilarityWrapper(args):
    # sys.stdout.write("created!\n")
    # sys.stdout.flush()
    # exit()
    global mysc0
    try:
        if mysc0 is None:  # type: ignore
            pass
    except:
        mysc0 = None
    if mysc0 is None:
        mysc0 = cv2.createShapeContextDistanceExtractor()
    contour = find_contour_png(args["batched_file"][1])
    if contour is None:
        # _in_pipe.send(1)
        return "", 99999999, []
    a = args["batched_file"][0], mysc0.computeDistance(args["query_contour"], contour), contour.tolist()
    # a=args["batched_file"][0],cv2.createHausdorffDistanceExtractor().computeDistance(args["query_contour"], contour),contour.tolist()
    return a
    # a = calculateSimilarity(cv2.createShapeContextDistanceExtractor(),args["query_contour"],args["batched_file"])
    # # args["pbar"].update()
    # return a


def getHW(integer):
    start = int(np.sqrt(integer))
    factor = integer / start
    while not is_integer(factor):
        start += 1
        factor = integer / start
    return int(factor), start


def is_integer(number):
    if int(number) == number:
        return True
    else:
        return False


def pad_img_to_square(original_image: Image.Image, padding_extra=0):
    width, height = original_image.size

    # if height == width:
    #     return np.asarray(original_image.resize(size_,Image.LANCZOS)).copy()

    if height > width:
        padding = (height - width) // 2
        new_size = (height + padding_extra * 2, height + padding_extra * 2)
    else:
        padding = (width - height) // 2
        new_size = (width + padding_extra * 2, width + padding_extra * 2)

    new_image = Image.new("RGBA", new_size, (0, 0, 0, 0))  # type: ignore

    if height > width:
        new_image.paste(original_image, (padding + padding_extra, 0 + padding_extra))
    else:
        new_image.paste(original_image, (0 + padding_extra, padding + padding_extra))
    a = np.asarray(new_image.resize(size_, Image.LANCZOS)).copy()
    return a


def pad_query_img_to_square(original_image: Image.Image, padding_extra=0):
    width, height = original_image.size

    # if height == width:
    #     return np.asarray(original_image.resize(size_,Image.LANCZOS)).copy()

    if height > width:
        padding = (height - width) // 2
        new_size = (height + padding_extra * 2, height + padding_extra * 2)
    else:
        padding = (width - height) // 2
        new_size = (width + padding_extra * 2, width + padding_extra * 2)

    new_image = Image.new("RGBA", new_size, (0, 0, 0, 0))  # type: ignore

    if height > width:
        new_image.paste(original_image, (padding + padding_extra, 0 + padding_extra))
    else:
        new_image.paste(original_image, (0 + padding_extra, padding + padding_extra))
    a = np.asarray(new_image.resize(size_, Image.LANCZOS)).copy()
    return a


def pad_query_img(query_img, bbox, padding=5):
    min_x = min(bbox[0][0], bbox[1][0])
    max_x = max(bbox[0][0], bbox[1][0])
    min_y = min(bbox[0][1], bbox[1][1])
    max_y = max(bbox[0][1], bbox[1][1])
    min_X = int(len(query_img) * min_x - padding)
    max_X = int(len(query_img) * max_x + padding)
    min_Y = int(len(query_img) * min_y - padding)
    max_Y = int(len(query_img) * max_y + padding)
    min_X = max(0, min_X)
    max_X = min(len(query_img[0]) - 1, max_X)
    min_Y = max(0, min_Y)
    max_Y = min(len(query_img) - 1, max_Y)
    cv2.imwrite("query_img_pad1.png", query_img[min_Y:max_Y, min_X:max_X])
    return query_img[min_Y:max_Y, min_X:max_X]


# _out_pipe, _in_pipe = Pipe(True)


def getReferenceRatios(png):
    img_gray = png[:, :, 3]
    contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    target_contour = max(contours, key=cv2.contourArea)
    (center, (width, height), angle) = cv2.minAreaRect(target_contour)
    return width / height


def getMostSimilarImages(
    path_prefix,
    query_img,
    bbox,
    n=16,
    max_sample=99999,
    batchsize=16,
    multi_process=False,
    debug=False,
    tmp_dir="tmp",
    simple=False,
):
    """
    # Parameters
    * path_prefix: 候选图片所在文件夹
    * query_img: 需要进行比对的图片的路径，**注意图片背景必须为透明**
    * n: 返回值取distance最小的前n个
    * max_sample: 从path_prefix中取前max_sample个作为比对样本，如果大于总体就取总体
    * batchsize: 仅当`multi_process=True`时生效，线程池大小
    * multi_process: 启用/禁用多线程

    # Return Value
    * `[[filename1, distance1], [filename2, distance2], ..., [filenamen, distancen]]`
    * 其中filename仅为纯文件名，不包含path_prefix，distance越小表明该图与query_img越相似
    """
    files = [os.path.join(path_prefix, file) for file in os.listdir(path_prefix)]
    if max_sample < len(files):
        random.shuffle(files)
        files = files[:max_sample]
    global pngs
    pngs = globals().get("pngs", {})
    try:
        if pngs is None:  # type: ignore
            pass
    except:
        pngs = {}
    if pngs == {}:
        for file in files:
            pngs[file] = np.asarray(Image.open(file))  # type: ignore
    if simple:
        min_x = min(bbox[0][0], bbox[1][0])
        max_x = max(bbox[0][0], bbox[1][0])
        min_y = min(bbox[0][1], bbox[1][1])
        max_y = max(bbox[0][1], bbox[1][1])
        query_ratio = abs((max_y - min_y) / (max_x - min_x))
        ratios = {}
        for png_name in pngs:
            ratios[png_name] = getReferenceRatios(pngs[png_name])  # type: ignore
        candidates = [[png_name, abs(query_ratio - ratios[png_name])] for png_name in ratios]
        candidates.sort(key=lambda x: x[1])
        return [[candidates[i][0], candidates[i][1]] for i in range(n)]
    # batched_files=[]
    # while total_sample<max_sample:
    #     total_sample+=batchsize
    #     if total_sample<=max_sample:
    #         batched_files.append(files[total_sample-batchsize:total_sample])
    #     elif total_sample-batchsize<max_sample:
    #         batched_files.append(files[total_sample-batchsize:max_sample])
    # if multi_process:
    #     proc_pool=Pool(batchsize)
    distances = []
    contours = []
    candidates = []
    query_contour = find_contour(pad_query_img(query_img, bbox))
    # query_contour=find_contour(pad_query_img_to_square(Image.fromarray(pad_query_img(query_img,bbox),mode="RGBA")))
    if type(query_contour) == str:
        return [[f, 0] for f in random.choices(files, k=n)]
    if debug:
        plt.figure(figsize=(10, 10))
        if type(query_img) == str:
            img_rgb = cv2.imread(query_img, cv2.IMREAD_UNCHANGED)
        else:
            # img_rgb=pad_img_to_square(Image.fromarray(query_img,bbox))
            img_rgb = pad_query_img_to_square(Image.fromarray(pad_query_img(query_img, bbox), mode="RGBA"))
        cv2.drawContours(img_rgb, [query_contour], 0, (0, 0, 255, 255), thickness=3)
        plt.imshow(img_rgb)
        # show_mask(query_mask[0], plt.gca())
        plt.axis("off")
        # plt.show()
        plt.savefig("query_img.png")
    # for i,batched_file in enumerate(tqdm(files)):
    #     distance,contour=calculateSimilarity(mysc,query_contour,batched_file) # type: ignore
    #     distances.append(distance)
    #     contours.append(contour)
    # return
    if multi_process:
        # pbar=tqdm(total=len(pngs))
        # pbar.set_description("calculateSimilarity")
        files_wrapper = [
            {
                # "pipe":(_out_pipe,_in_pipe),
                "query_contour": query_contour,
                "batched_file": file,
            }
            for file in zip(pngs.keys(), pngs.values())  # type: ignore
        ]
        res = iterate_wrapper(
            calculateSimilarityWrapper,
            files_wrapper,
            output=None,
            num_workers=batchsize,
            run_name="getMostSimilarImages",
            bar=True,
            tmp_dir=tmp_dir,
        )
        candidates = res
    else:
        # mysc=cv2.createShapeContextDistanceExtractor()
        mysc = cv2.createHausdorffDistanceExtractor()
        for i, batched_file in enumerate(tqdm(files)):
            filename, distance, contour = calculateSimilarity(mysc, query_contour, batched_file)  # type: ignore
            candidates.append([filename, distance, contour])
    candidates.sort(key=lambda x: x[1])
    # print("sort done!")
    if debug:
        plt.figure(figsize=(15, 15))
        H, W = getHW(n)
        for i in range(n):
            ax = plt.subplot(H, W, i + 1)
            imageax = cv2.imread(candidates[i][0], cv2.IMREAD_UNCHANGED)
            imageax = cv2.cvtColor(imageax, cv2.COLOR_RGBA2RGB)
            imageax = pad_img_to_square(Image.fromarray(imageax))
            cv2.drawContours(imageax, [np.array(candidates[i][2])], 0, (0, 0, 255))
            ax.text(0, 0, str(candidates[i][1]))
            ax.imshow(imageax)
        # plt.show()
        plt.savefig("result.png")
    return [[candidates[i][0], candidates[i][1]] for i in range(n)]


# if __name__=="__main__":
#     path_prefix="./pics/"

#     options = [
#         "Fusulina_huntensis_1_3.png",
#         "Fusulinella_devexa_1_11.png",
#         "Fusulinella_famula_1_1.png",
#         "Pseudofusulina_modesta_1_1.png",
#         "Fusulinella_clarki_1_1.png",
#         "Fusulinella_clarki_1_6.png",
#         "Fusulina_boonensis_1_1.png",
#         "Fusulinella_cabezasensis_1_5.png",
#         "Chusenella_absidata_1_2.png",
#         "Rugosofusulina_yingebulakensis_1_1.png",
#     ]
#     options=[path_prefix+img for img in options]
#     indice = 0
#     query_image=cv2.imread(options[indice],cv2.IMREAD_UNCHANGED)
#     getMostSimilarImages(path_prefix,query_image,None,batchsize=128,multi_process=True,debug=False)
