from data.caption.caption_2nd.params import *
from data.caption.caption_2nd.base import BaseFeature
import json
from data.rule.shapes import GSRule

class Tunnel(BaseFeature):
    def __init__(self, rule000, visible_chomata_idx, chomata_whs_relative, tunnel_angles=[]):
        self.tunnel_angles = [round(x, 0) for x in tunnel_angles]
        self.threshold = 10
        self.rule = rule000
        self.visible_chomata_idx = visible_chomata_idx
        self.chomata_whs_relative=chomata_whs_relative

    def getTunnelHeight(self):
        tunnel_heights=[self.chomata_whs_relative[k][1] for k in self.chomata_whs_relative]
        res = self.standardRangeFilter(chomata_height_classes, sum(tunnel_heights)/len(tunnel_heights))
        if res=="moderate":
            res="height moderate"
        return res

    def genTunnelFeatures(self):
        feat = ""
        angles_classes = []
        for i in range(len(self.tunnel_angles)):
            for k in tunnel_angle_classes.keys():
                if (
                    abs(self.tunnel_angles[i]) >= tunnel_angle_classes[k][0]
                    and abs(self.tunnel_angles[i]) <= tunnel_angle_classes[k][1]
                ):
                    angles_classes.append(k)
                    break
        tunnel_height = self.getTunnelHeight()
        if angles_classes[0] == "narrow" and angles_classes[-1] == "narrow":
            feat = f"Tunnels {tunnel_height}, narrow. "
        elif angles_classes[0] == "narrow" and angles_classes[-1] == "broad":
            feat = f"Tunnels {tunnel_height} and narrow in inner volutions, but broad in outer volutions. "
        elif angles_classes[0] == "broad" and angles_classes[-1] == "broad":
            feat = f"Tunnels {tunnel_height}, broad. "
        elif angles_classes[0] == "moderate" and angles_classes[-1] == "broad":
            feat = f"Tunnels {tunnel_height}, broad in outer volutions and moderate in inner volutions. "
        elif angles_classes[0] == "narrow" and angles_classes[-1] == "moderate":
            feat = f"Tunnels {tunnel_height}, narrow in inner volutions and moderate in outer volutions. "
        elif angles_classes[0] == "moderate" and angles_classes[-1] == "narrow":
            feat = f"Tunnels {tunnel_height}, narrower in outer volutions compared to inner volutions. "
        elif angles_classes[0] == "broad" and (angles_classes[-1] == "moderate" or angles_classes[-1] == "narrow"):
            feat = f"Tunnels {tunnel_height}, broader in inner volutions compared to outer volutions. "
        else:
            feat = f"Tunnels {tunnel_height}, width moderate. "
        return feat

    def genTunnelAngleDescription(self):
        tunnel_map = ""
        for i in range(len(self.visible_chomata_idx)):
            if len(self.visible_chomata_idx) == 1:
                tunnel_map += ordinal_numbers[self.visible_chomata_idx[i]]
            elif i != len(self.visible_chomata_idx) - 1:
                tunnel_map += ordinal_numbers[self.visible_chomata_idx[i]]
                tunnel_map += ", "
            else:
                tunnel_map += "and {th}".format(th=ordinal_numbers[self.visible_chomata_idx[i]])
        txt = "Tunnel angles of {ths} volutions measure ".format(ths=tunnel_map)
        for i in range(len(self.tunnel_angles)):
            if len(self.tunnel_angles) == 1:
                txt += "{:.0f} degrees. ".format(self.tunnel_angles[i])
            elif i != len(self.tunnel_angles) - 1:
                txt += "{:.0f}".format(self.tunnel_angles[i])
                txt += ", "
            else:
                txt += "and {:.0f} degrees, respectively. ".format(self.tunnel_angles[i])
        return txt

    def genUserInput(self):
        tagged=[]
        txt = "<tunnel shape>"
        txt += self.genTunnelFeatures()
        txt+="</tunnel shape>"
        tagged.append(txt)
        txt = f"<tunnel angle>{self.genTunnelAngleDescription()}</tunnel angle>"
        tagged.append(txt)
        return tagged

    def genInput(self):
        txt = "tunnel angles: "
        for i in range(len(self.tunnel_angles)):
            if self.visible_chomata_idx[i] == 0:
                txt += "{angle} in the 1st volution".format(angle=self.tunnel_angles[i])
            elif self.visible_chomata_idx[i] + i == 1:
                txt += "{angle} in the 2nd volution".format(angle=self.tunnel_angles[i])
            elif self.visible_chomata_idx[i] + i == 2:
                txt += "{angle} in the 3rd volution".format(angle=self.tunnel_angles[i])
            else:
                txt += "{angle} in the {idx}th volution".format(
                    angle=self.tunnel_angles[i], idx=self.visible_chomata_idx[i] + 1
                )
            if i != len(self.tunnel_angles) - 1:
                txt += ", "
            else:
                txt += ".\n"
        return txt
