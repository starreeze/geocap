from data.caption_2nd.params import *
from data.caption_2nd.base import BaseFeature


class Tunnel(BaseFeature):
    def __init__(self, tunnel_start_idx, tunnel_angles=[]):
        self.tunnel_angles = [round(x, 0) for x in tunnel_angles]
        self.threshold = 10
        self.tunnel_start_idx = tunnel_start_idx

    def genTunnelFeatures(self):
        feat = ""
        angles_classes = []
        for i in range(len(self.tunnel_angles)):
            for k in tunnel_angle_classes.keys():
                if (
                    self.tunnel_angles[i] >= tunnel_angle_classes[k][0]
                    and self.tunnel_angles[i] <= tunnel_angle_classes[k][1]
                ):
                    angles_classes.append(k)
                    break
        if angles_classes[0] == "narrow" and angles_classes[-1] == "narrow":
            feat = "Tunnels low, narrow. "
        elif angles_classes[0] == "narrow" and angles_classes[-1] == "broad":
            feat = "Tunnels low and narrow in inner volutions, and broad in outer volutions. "
        elif angles_classes[0] == "broad" and angles_classes[-1] == "broad":
            feat = "Tunnels low, broad. "
        elif angles_classes[0] == "" and angles_classes[-1] == "broad":
            feat = "Tunnels broad in outer volutions. "
        elif angles_classes[0] == "narrow" and angles_classes[-1] == "":
            feat = "Tunnels narrow in inner volutions. "
        return feat

    def genTunnelAngleDescription(self):
        txt = "Tunnel angles of {start} to {end} volutions measure ".format(
            start=ordinal_numbers[self.tunnel_start_idx],
            end=ordinal_numbers[len(self.tunnel_angles) - 1 + self.tunnel_start_idx],
        )
        for i in range(len(self.tunnel_angles)):
            if i != len(self.tunnel_angles) - 1:
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
