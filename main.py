from cv import *

if __name__ == '__main__':

    comvis = ComputerVision("static/images.jpeg")
    output = comvis.whitening()
    ComputerVision.show_image(output)
    #ComputerVision.plt_image(output,'gray')
    output2 = comvis.history_equalization()
    ComputerVision.plt_image(output2,'gray')
