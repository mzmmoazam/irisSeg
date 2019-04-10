# irisSeg


Segementation of iris and pupil.

## Daugman algorithm:

![image](irisSeg/Data/DaughmanOperation.jpg)

`where I(x,y) is the eye image, r is the radius to searches over the image (x,y), G(r) is a Gaussian smoothing function.
The algorithm starts to search from the pupil, in order to detect the changing of maximum pixel values
(partial derivative).`

# Installation
    pip install irisSeq

# Usage

    from irisSeg import irisSeg
    import matplotlib.pyplot as plt
    #
    # you can also view using the argument in irisSeq function
    #
    coord_iris, coord_pupil, output_image = irisSeg('UBIRIS_200_150_R/Sessao_1/1/Img_1_1_1.jpg', 40, 70)
    print(coord_iris)
    print(coord_pupil)
    plt.imshow(output_image)
    plt.show()

![image](irisSeg/Data/sample_img.jpg)

![image](irisSeg/Data/segemented_img.png)

# References

1. `https://www.diva-portal.org/smash/get/diva2:831173/FULLTEXT01.pdf`
2. `https://uk.mathworks.com/matlabcentral/fileexchange/15652-iris-segmentation-using-daugman-s-integrodifferential-operator`