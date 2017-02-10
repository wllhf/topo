import img_toolbox.dataset as input_data
import img_toolbox.plot as tbx_plt

GRID_SIZE = [20, 20]
PATCH_SIZE = [100, 100]
NCHANNELS = 3

PATH = "/home/mw/Workspace/topo_data/set_02_falc/"


data = input_data.img_data(PATH)
data.sample('train', 'class', GRID_SIZE, PATCH_SIZE, balanced=True, shuffle=False, one_hot=False)
imgs = data.load_image_data()
print data._samples.shape
print data._labels[-20:]
tbx_plt.plot_patches(imgs, data._samples[-20:, :], PATCH_SIZE)
