from data.caption.caption_2nd.params import *
from data.caption.caption_2nd.base import BaseFeature


class Septa(BaseFeature):
    def __init__(self, septa_folds):
        self.septa_folds = septa_folds
        self.txt = ""
        self.txt2 = ""
        self.refineSeptaFolds()
        self.genSeptaNum()
        self.genSeptaDescription()

    def refineSeptaFolds(self):
        refined_septa_folds = {}
        current_volution = 1
        for septa in self.septa_folds:
            if "special_info" in septa:
                current_volution = int(septa["special_info"][len("septa of volution ") :])
            if current_volution not in refined_septa_folds:
                refined_septa_folds[current_volution] = {}
            if septa["type"] not in refined_septa_folds[current_volution]:
                refined_septa_folds[current_volution][septa["type"]] = []
            refined_septa_folds[current_volution][septa["type"]].append(septa)
        self.refined_septa_folds = refined_septa_folds

    def genSeptaNum(self):
        if len(self.septa_folds) == 0:
            return
        txt = "The septal counts for the {start} to {end} volution are ".format(
            start=ordinal_numbers[min(self.refined_septa_folds.keys())],
            end=ordinal_numbers[max(self.refined_septa_folds.keys())],
        )
        txt2 = "the septal counts: "
        idx = 0
        for i in self.refined_septa_folds:
            total = 0
            for v in self.refined_septa_folds[i].values():
                total += len(v)
            if idx != len(self.refined_septa_folds) - 1:
                txt += str(total)
                txt += ", "
                txt2 += str(total)
                txt2 += ", "
            else:
                txt += "and {n}, respectively. ".format(n=total)
                txt2 += str(total)
                txt2 += ".\n"
            idx += 1
        self.txt += txt
        self.txt2 += txt2

    def genSeptaDescription(self):
        if len(self.septa_folds) == 0:
            return
        inner = [0, 0, 0]
        outer = [0, 0, 0]
        tags = ["polygon", "ellipse", "curves"]
        for k in self.refined_septa_folds:
            if k <= len(self.refined_septa_folds) / 2:
                for i in range(len(tags)):
                    if tags[i] in self.refined_septa_folds[k]:
                        inner[i] += len(self.refined_septa_folds[k][tags[i]])
            else:
                for i in range(len(tags)):
                    if tags[i] in self.refined_septa_folds[k]:
                        outer[i] += len(self.refined_septa_folds[k][tags[i]])
        inner_r = [k / sum(inner) for k in inner]
        outer_r = [k / sum(outer) for k in outer]
        adj1 = self.overridedLambdaFilter(septa_shape_classes, inner_r)
        adj2 = self.overridedLambdaFilter(septa_shape_classes, outer_r)
        if adj1 != adj2:
            txt = "Septa {adj1} in the inner whorls and {adj2} in the outer whorls. ".format(
                adj1=adj1, adj2=adj2
            )
        else:
            txt = "Septa {adj}. ".format(adj=adj1)
        self.txt += txt

    def genUserInput(self):
        return self.txt

    def genInput(self):
        return self.txt2
