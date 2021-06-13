import colorsys
import copy
import numpy as np 
import torchvision
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.axis('off')

def plot_heatmap_2d(df, filename=None, ax=None, cax=None, labels_on=True, **kwargs):
  cmap = plt.cm.viridis # copy.copy(sns.color_palette("coolwarm", as_cmap=True))
  # cmap.set_bad("silver")
  ax = sns.heatmap(df, cmap=cmap, cbar=labels_on, ax=ax, cbar_ax=cax, cbar_kws={"ticks":[0.0,0.02,0.04], 'label':'MSE'}, **kwargs)
  if cax is not None:
    cax.tick_params(labelsize=50) 
  ax.set_xticks([])
  ax.set_yticks([])
  if filename is not None:
    fig = ax.get_figure() 
    fig.tight_layout()
    fig.savefig(filename)

def plot_polar_barchart(x, filename=None, ax=None, vmin=None, vmax=None, xticks=None, yticks=None, labels_on=True):
  # Compute pie slices
  N = len(x)
  theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
  radii = x
  width = 1.8 * np.pi / N * np.ones(N) 
  colors = plt.cm.viridis( 0.5 * radii / max(radii))

  if filename is not None:
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, polar=True)
  else:
    assert ax is not None, "filename and ax cannot both be None"
  ax.bar(theta, radii, width=width, bottom=0.0, color=colors, alpha=1.0)
  ax.grid(linewidth=4., linestyle="--")
  ax.spines['polar'].set_linewidth(4.)
  ax.set_ylim(bottom=vmin, top=vmax)
  ax.set_theta_zero_location('N')
  if not labels_on:
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
  if xticks is not None:
    ax.set_xticks(np.array(xticks))
  if yticks is not None:
    ax.set_yticks(np.array(yticks))
  ax.tick_params(axis='both', which='major', labelsize=50)
  if filename is not None:
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def visualize_2d_vector(z, v, img_size=64, filename='test.png'):
  batch_size = z.size()[0]
  z = z.detach().cpu().numpy()
  v = v.detach().cpu().numpy()
  cos = np.cos((2*np.pi)/4*v).round()
  sin = np.sin((2*np.pi)/4*v).round()
  dz = np.concatenate([sin, cos], axis=1)
  
  fig = plt.figure(figsize=(3.2*batch_size, 3), facecolor='xkcd:salmon')
  for i in range(batch_size):
    ax = fig.add_subplot(1, batch_size, i+1)
    ax.set_xlim(0, img_size)
    ax.set_ylim(0, img_size)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    #ax.axis("off")
    ax.quiver(z[i, 0], z[i, 1], dz[i, 0], dz[i, 1], scale=10)
  fig.savefig(filename)
  plt.close(fig)
  
  
def visualize_images(imgs, filename, nrow=None):
  if nrow is None:
    nrow = imgs.size()[0]
  torchvision.utils.save_image(imgs, filename, nrow=nrow, pad_value=1)
    
def visualize_latents(latents, filename):
  batch_size, _, _, num_channels = latents.shape
  fig = plt.figure(figsize=(3*num_channels, 3))
  for i in range(num_channels):
    ax = fig.add_subplot(1, num_channels, i+1)
    ax.axis("off")
    ax.imshow(latents[0, :, :, i])
  fig.savefig(filename, bbox_inches='tight')
  plt.close(fig)