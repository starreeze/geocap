from data.caption.caption_2nd.params import *
from data.caption.caption_2nd.base import BaseFeature


class Deposit(BaseFeature):
    def __init__(self, axial_filling, volution_num):
        self.axial_filling = axial_filling
        self.volution_num = volution_num

    def getDepositDevelopment(self) -> str:
        min_start_volution = 999
        max_end_volution = 0
        for c in self.axial_filling:
            if c["type"] == "main":
                if c["end_volution"] > max_end_volution:
                    max_end_volution = c["end_volution"]
                if c["start_volution"] < min_start_volution:
                    min_start_volution = c["start_volution"]
        if len(self.axial_filling) == 0:
            min_start_volution = 0
        return self.standardRangeFilter(
            deposit_development_classes, (max_end_volution - min_start_volution) / self.volution_num
        )

    def genUserInput(self):
        dev = self.getDepositDevelopment()
        # if dev == "":
        #     return ""
        txt = "Axial filling {block}. ".format(block=dev)
        return [f"<axial filling>{txt}</axial filling>"]
