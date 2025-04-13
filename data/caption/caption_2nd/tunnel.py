import json

from data.caption.caption_2nd.base import BaseFeature
from data.caption.caption_2nd.params import *


class Tunnel(BaseFeature):
    def __init__(self, rule000, visible_chomata_idx, tunnel_angles=[]):
        self.tunnel_angles = [round(x, 0) for x in tunnel_angles]
        self.threshold = 10
        self.rule = rule000
        self.visible_chomata_idx = visible_chomata_idx

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
        if len(angles_classes) == 0:
            with open("dataset/error2.json", "w") as f:
                json.dump(self.rule, f, indent=4)
            exit()
        if angles_classes[0] == "narrow" and angles_classes[-1] == "narrow":
            feat = "Tunnels low, narrow. "
        elif angles_classes[0] == "narrow" and angles_classes[-1] == "broad":
            feat = "Tunnels low and narrow in inner volutions, and broad in outer volutions. "
        elif angles_classes[0] == "broad" and angles_classes[-1] == "broad":
            feat = "Tunnels low, broad. "
        elif angles_classes[0] == "moderate" and angles_classes[-1] == "broad":
            feat = "Tunnels broad in outer volutions and moderate in inner volutions. "
        elif angles_classes[0] == "narrow" and angles_classes[-1] == "moderate":
            feat = "Tunnels narrow in inner volutions and moderate in outer volutions. "
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
        txt = ""
        txt += self.genTunnelFeatures()
        txt += self.genTunnelAngleDescription()
        return txt

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
