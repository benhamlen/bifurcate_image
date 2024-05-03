import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import zoom
import sys
from skimage.transform import resize
from tqdm import tqdm

VISUALIZE = False
DEBUG_PRINT = False

def display_matrices(original_matrix, transformed_matrix):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Display the original matrix
    axes[0].imshow(original_matrix, cmap='viridis', origin='upper', aspect='auto')
    axes[0].set_title('Original Matrix')
    axes[0].axis('off')

    # Display the transformed matrix
    axes[1].imshow(transformed_matrix, cmap='viridis', origin='upper', aspect='auto')
    axes[1].set_title('Transformed Matrix')
    axes[1].axis('off')

    plt.show()

def logistic_map(x, r):
    return r * x * (1 - x)

def logistic_bifurcation(matrix):
    # Define the logistic map parameters
    r = 3.8  # You can experiment with different values of r

    # Calculate the index for the logistic bifurcation
    logistic_ratio_index = int(matrix.shape[1] * logistic_map(0.5, r))

    # Apply the logistic bifurcation
    transformed_matrix = apply_logistic_bifurcation(matrix, logistic_ratio_index)

    # Display the original and transformed matrices side by side
    display_matrices(matrix, transformed_matrix)

    # Save the matrices as TIF images
    # cv2.imwrite('original_image.tif', matrix)
    # cv2.imwrite('transformed_image.tif', transformed_matrix)

    return transformed_matrix

def apply_logistic_bifurcation(matrix, logistic_ratio_index):
    # Duplicate the contents at the logistic ratio index to both top and bottom
    top_and_bottom = matrix[:, logistic_ratio_index:logistic_ratio_index+1]

    # Concatenate the left, top_and_bottom, and right parts along the second axis
    transformed_matrix = np.concatenate((matrix[:, :logistic_ratio_index], top_and_bottom, matrix[:, logistic_ratio_index:]), axis=1)

    return transformed_matrix

def logistic(r, x):
    return r * x * (1 - x)

def plotBifurcationDiagram(img_shape):
    HEIGHT = img_shape[0]
    RMAX = 4.0
    RMIN = 2.5
    XMIN = 0
    XMAX = 1.0
    # n = 10000
    n = img_shape[1]

    YMIN = -1
    YMAX = 2

    output_map = np.zeros(img_shape)

    r = np.linspace(RMIN, RMAX, n)
    iterations = 1000
    last = 100
    x = 1e-5 * np.ones(n)
    lyapunov = np.zeros(n)


    col_inds = n*((RMAX-r)/(RMAX-RMIN))
    row_inds = (HEIGHT*((XMAX-x)/(XMAX-XMIN)))

    bifurcation_map = np.zeros((n,HEIGHT))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9),
                                sharex=True)
    for i in range(iterations):
        x = logistic(r, x)

        # print(col_inds.shape)
        # print(row_inds.shape)

        # We compute the partial sum of the
        # Lyapunov exponent.
        lyapunov += np.log(abs(r - 2 * r * x))

        # We display the bifurcation diagram.
        if i >= (iterations - last):
            ax1.plot(r, x, ',k', alpha=.25)
            output_cols = np.array((r-RMIN)/(RMAX-RMIN) * (img_shape[1]-1)).clip(0,img_shape[1]-1).astype(np.int64)
            output_rows = np.array((1 - ((x-YMIN)/(YMAX-YMIN))) * (img_shape[0]-1)).clip(0,img_shape[0]-1).astype(np.int64)
            output_map[output_rows, output_cols] = 1

    ax1.set_xlim(RMIN, RMAX)
    ax1.set_title("Bifurcation diagram")

    # We display the Lyapunov exponent.
    # Horizontal line.
    ax2.axhline(0, color='k', lw=.5, alpha=.5)
    # Negative Lyapunov exponent.
    ax2.plot(r[lyapunov < 0],
            lyapunov[lyapunov < 0] / iterations,
            '.k', alpha=.5, ms=.5)
    # Positive Lyapunov exponent.
    ax2.plot(r[lyapunov >= 0],
            lyapunov[lyapunov >= 0] / iterations,
            '.r', alpha=.5, ms=.5)
    ax2.set_xlim(RMIN, RMAX)
    ax2.set_ylim(YMIN, YMAX)
    ax2.set_title("Lyapunov exponent")
    plt.tight_layout()
    plt.savefig("bifurcation_and_lyapunov_diagrams.png", dpi=300)
    # plt.show()
    plt.close()

    return output_map


def bifurcate_image(img: np.array, bif_map: np.array):
    
    assert(type(img) == np.ndarray 
       and type(bif_map) == np.ndarray
       and img.shape == bif_map.shape)
    
    output_map = np.zeros_like(img)
    
    PREVIEW_SCALE = 0.25

    # cv2.namedWindow("bif_map")

    preview_output_map = np.zeros_like(output_map)
    preview_bif_map = np.copy(bif_map)

    preview_output_map = np.array(zoom(preview_output_map, PREVIEW_SCALE))
    preview_bif_map = np.array(zoom(preview_bif_map, PREVIEW_SCALE))
    
    if VISUALIZE:
        cv2.namedWindow("output_map")
        cv2.imshow("bif_map", preview_bif_map)
        # cv2.imshow("bif_map_full", bif_map)

    # iterate over each column and stamp scaled down copies of input image
    # column according to number of bifurcations
    col_resize_factor_last = 1
    for col_ind in tqdm(range(output_map.shape[1])):
        if DEBUG_PRINT:
            tqdm.write("----------")
            tqdm.write(f"column {col_ind}:")

        # make a copy of the column in img
        col = img[:,col_ind]
        col_bif_map = bif_map[:,col_ind]

        # skip this columnn if no non-zero elements
        if np.any(col_bif_map):

            # get the number of non-zero elements in this col in bif_map
            nonzero_inds = np.where(col_bif_map > 0)
            if DEBUG_PRINT: 
                tqdm.write(f"found {len(nonzero_inds)} non-zero indecies in column {col_ind}: {nonzero_inds}")
            if len(nonzero_inds) > 1:
                tqdm.write(f"nonzero_inds has length of {len(nonzero_inds)}, aborting program to investigate...")
                sys.exit(99)
            else:
                nonzero_inds = nonzero_inds[0]

            # low-pass filter resize factor 
            col_resize_factor = np.clip(1/(len(nonzero_inds)**2), 0.01, np.inf)
            alpha = 0.0000025 + ((col_ind/output_map.shape[1])**5)*0.1
            col_resize_factor_filt = col_resize_factor * alpha + col_resize_factor_last * (1-alpha)
            col_resize_factor_last = col_resize_factor_filt
            col = resize(col, (int(col.shape[0]*col_resize_factor_filt),), order=3)

            # stamp copies of copied col centered around points in bif_map in this col
            for ind in nonzero_inds:
                
                # calculate bounds where new col will be pasted (include clipping for edges of frame)
                rmin = ind - int(col.shape[0]/2)
                rmax = ind + int(col.shape[0]/2)
                rmin_clipped = np.clip(rmin, 0, img.shape[0])
                rmax_clipped = np.clip(rmax, 0, img.shape[0])
                amount_clipped_top = rmax - rmax_clipped
                amount_clipped_bottom = rmin_clipped - rmin

                # update col with clipping
                # if amount_clipped_top:
                #     col = col[:-amount_clipped_top]
                # elif amount_clipped_bottom:
                #     col = col[amount_clipped_bottom:]

                # assert(col.shape[0] == 1/len(nonzero_inds) * (rmax_clipped - rmin_clipped))

                # reshape col to new trimmed size
                # tqdm.write(f"col shape before resizing: {col.shape}")
                col = resize(col, (rmax_clipped - rmin_clipped,), order=3)
                # tqdm.write(f"col shape after resizing: {col.shape}")

                output_map[rmin_clipped:rmax_clipped, col_ind] += (col*255).astype(np.uint8)
                # preview_output_map[:,int(col_ind*PREVIEW_SCALE)] = 255*zoom(output_map[:,col_ind], PREVIEW_SCALE)

            if VISUALIZE:
                # show previews
                # preview_output_map_withbar = np.copy(preview_output_map)
                # preview_output_map_withbar[:,col_ind] = 255
                # cv2.imshow("output_map", preview_output_map_withbar)

                # bif_map_withbar = np.copy(bif_map)
                # bif_map_withbar[:,col_ind] = 255
                # cv2.imshow("bif_map_full", bif_map_withbar)

                # preview_bif_map_withbar = np.copy(preview_bif_map)
                # preview_bif_map_withbar[:,] = 255
                # cv2.imshow("bif_map", preview_bif_map_withbar)

                cv2.imshow("output_map", output_map)

                cv2.waitKey(1)
    else:
        if DEBUG_PRINT:
            tqdm.write(f"found no non-zero indecies in column {col_ind}")
    
    print("done.")

    if VISUALIZE:
        cv2.destroyAllWindows()

    return output_map

# Example usage:
# logistic_bifurcation(tif_matrix)

# import image
tif_path = "C:/Users/Hamle/Desktop/bifurcations.tif"
tif_matrix = cv2.imread(tif_path, cv2.IMREAD_UNCHANGED)
tif_matrix = cv2.bitwise_not(tif_matrix)
tif_matrx = np.array(tif_matrix)

if VISUALIZE:
    plt.imshow(tif_matrix)
    plt.show()
    plt.close()

# resize image if wanted 

# get image resolution
img_shape = tif_matrix.shape
print(img_shape)

# calculate binary bifurcation map at image resolution
map = plotBifurcationDiagram(img_shape)
print(f"map shape: {map.shape}")
# plt.imshow(map)
# plt.show()
# plt.close()

# pass image and binary map to function to bifurcate image
img_bif = bifurcate_image(tif_matrix, map)
print(f"map shape: {img_bif.shape}")
cv2.imwrite("bifurcated_img.png", img_bif)
plt.imshow(img_bif)
# plt.savefig("bifurcated_image.png", dpi=300)
plt.show()
