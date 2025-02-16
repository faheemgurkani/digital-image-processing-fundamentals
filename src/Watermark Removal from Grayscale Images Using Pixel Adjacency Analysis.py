import numpy as np
import os
import cv2
import matplotlib.pyplot as plt



class WatermarkRemover:
    """
    A class to detect and remove a visible watermark from a grayscale image
    using only pixel adjacency analysis and spatial filtering.
    
    Steps:
        1. Read/Store the image (Grayscale).
        2. Detect the watermark by thresholding + connected components.
        3. Create a binary mask isolating the watermark.
        4. Remove the watermark by filling its region with local neighbor averages.
        5. Smooth and refine the filled region using spatial filtering.
        6. Provide or save the final, watermark-free image.
    """
    
    def __init__(self, image_path, threshold_value=180, kernel_size=3, fill_iterations=1):
        """
        Initializes the WatermarkRemover class.

        Args:
            image_path (str): Path to the input grayscale image.
            threshold_value (int): Intensity threshold to detect the watermark region.
            kernel_size (int): Size of the morphological operation kernel.
            fill_iterations (int): Number of passes to fill the watermark region (more passes may improve results).
        """
        # Reading the grayscale image
        self.image_path = image_path
        
        self.original_image = self.read_image()

        filename = os.path.basename(self.image_path)

        self.image_name, ext = os.path.splitext(filename)
        
        self.threshold_value = threshold_value
        self.kernel_size = kernel_size
        self.fill_iterations = fill_iterations
        
        # Internal placeholders for intermediate results
        self.binary_mask = None
        self.restored_image = None
        self.smoothed_image = None
        
        # Storing dimensions for convenience
        self.height, self.width = self.original_image.shape

    def read_image(self):
        """
        Reads the image in grayscale and displays it.
        Returns:
            image (numpy.ndarray): The grayscale image.
        """
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Image file '{self.image_path}' not found.")

        self.display_image(image, "Original Image")

        return image

    def display_image(self, image, title="Image"):
        """
        Displays an image using matplotlib.
        Args:
            image (numpy.ndarray): Image to display.
            title (str): Title for the displayed image.
        """
        plt.figure(figsize=(6,6))

        plt.imshow(image, cmap='gray')

        plt.title(title)

        plt.axis('off')

        plt.show()

    def detect_watermark(self):
        """
        Detects the watermark region using a combination of:
         - Simple thresholding (global or adaptive).
         - Connected components or morphological operations to refine the mask.
        
        Explanation:
            1. Thresholding separates the (likely brighter) watermark pixels from the background.
            2. A morphological operation (opening) helps remove noise or small irrelevant regions.
            3. We produce a binary mask where mask[y,x] = 1 if it's a watermark pixel, else 0.
        """
        # Thresholding the image
        # The threshold_value can be adjusted based on how bright/dark the watermark is.
        # _, thresh = cv2.threshold(
        #     self.original_image, self.threshold_value, 255, cv2.THRESH_BINARY
        # )
        _, thresh = cv2.threshold(self.original_image, self.threshold_value, 255, cv2.THRESH_BINARY_INV)

        
        # Refining the mask with morphological opening
        #    Opening = Erosion followed by Dilation; it removes small white noise (pecks).
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        refined_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        self.display_image(refined_mask, "Refined Mask")
        
        # Converting the refined mask to boolean form
        self.binary_mask = refined_mask > 0

        self.display_image(self.binary_mask, "Binary Mask")

    def remove_watermark(self):
        """
        Removes the watermark by filling the detected region with
        an average of neighboring pixel values (not in the watermark).

        Explanation:
            - We create a copy of the original image in float32 format (for easy averaging).
            - For each pixel in the mask, we look at the 8-connected neighbors.
            - If a neighbor is not part of the watermark, we gather its intensity for averaging.
            - We perform multiple iterations to gradually fill from outside in (if desired).
        """
        # If the mask was not created yet, detect the watermark first
        # if self.binary_mask is None:
        #     self.detect_watermark()   

        if not hasattr(self, 'binary_mask'):
            raise ValueError("Mask was not created yet, detect the watermark first")

        # Converting the original image to float for averaging
        restored = self.original_image.astype(np.float32).copy()
        
        # Coordinates of all pixels in the mask
        mask_points = np.argwhere(self.binary_mask)

        # Define 8-connected neighbor offsets
        neighbor_offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            ( 0, -1),          ( 0, 1),
            ( 1, -1), ( 1, 0), ( 1, 1)
        ]
        
        # Repeating the filling process for the specified number of iterations
        for _ in range(self.fill_iterations):
            
            for (y, x) in mask_points:
                valid_neighbors = []
            
                for (dy, dx) in neighbor_offsets:
                    ny, nx = y + dy, x + dx
            
                    # Checking image bounds
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        
                        # Only considering neighbor if it's not in the watermark
                        if not self.binary_mask[ny, nx]:
                            valid_neighbors.append(restored[ny, nx])
                
                # If we have valid neighbors, average them; else, fallback to original
                if len(valid_neighbors) > 0:
                    restored[y, x] = np.mean(valid_neighbors)
                else:
                    restored[y, x] = self.original_image[y, x]

        # Storing the intermediate restored result
        self.restored_image = restored

    def smooth_and_refine(self, method="median", ksize=9):
        """
        Applies a spatial filter (median or box blur) to blend the filled region with the surroundings.

        Args:
            method (str): "median" or "box" to specify which filter to use.
            ksize (int): Kernel size for smoothing.

        Explanation:
            - After filling the watermark region, edges might be abrupt.
            - A median or box (mean) filter helps smooth transitions.
            - The smoothing can be applied to the entire image or selectively to the mask boundary.
        """
        # if self.restored_image is None:
        #     self.remove_watermark()

        if not hasattr(self, 'remove_watermark'):
            raise ValueError("Watermark was not removed yet, run remove_watermark first")
        
        # Converting restored float image to uint8
        restored_uint8 = self.restored_image.astype(np.uint8)
        
        if method == "median":
            # Median filter is good at removing salt-and-pepper noise and smoothing edges gently
            self.smoothed_image = cv2.medianBlur(restored_uint8, ksize)
        elif method == "box":
            # Box filter (mean blur) for simpler smoothing
            self.smoothed_image = cv2.blur(restored_uint8, (ksize, ksize))
        else:
            raise ValueError("Unknown smoothing method. Use 'median' or 'box'.")

    def save_output(self, output_path):
        """
        Saves the final, watermark-removed image to disk.

        Args:
            output_path (str): Where to save the restored image.

        Explanation:
            - The final image (self.smoothed_image) should look natural
              with minimal artifacts or abrupt transitions.
        """
        if self.smoothed_image is None:
            # If smoothing wasn't called, just save the restored image (no smoothing)
            if self.restored_image is not None:
                to_save = self.restored_image.astype(np.uint8)
            else:
                raise ValueError("No image available to save. Please run remove_watermark() first.")
        else:
            to_save = self.smoothed_image

        self.display_image(to_save, "Restored Image")
        
        self.save_image(output_path, to_save)

    def save_image(self, save_path, to_save):
        """
        Saves the binarized image to the specified path.
        If the file already exists, it is deleted before saving the new image.
        Args:
            save_path (str): Path to save the processed image.
        """
        # Defining the target directory
        results_dir = os.path.join("../results")
        
        # Creating the directory if it does not exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Constructing the full path for saving the image
        full_save_path = os.path.join(results_dir, f"{save_path}{self.image_name}_restored.png")
        
        # Removing existing file if it exists
        if os.path.exists(full_save_path):
            os.remove(full_save_path)
        
        # Saving the image
        cv2.imwrite(full_save_path, to_save)
        
        print(f"Image saved at {full_save_path}")

    def get_result(self):
        """
        Returns the final smoothed image in NumPy array format (H x W).
        
        Explanation:
            - Useful if you want to process the result further in memory.
        """
        if self.smoothed_image is not None:
            return self.smoothed_image
        elif self.restored_image is not None:
            return self.restored_image.astype(np.uint8)
        else:
            raise ValueError("No result yet. Please run remove_watermark() and/or smooth_and_refine().")
        
    def process_pipeline(self, save_path):
        """
        Runs the complete watermark removal pipeline:
            1. Detect watermark.
            2. Remove watermark.
            3. Smooth and refine the result.
            4. Save the final image.
        """
        self.detect_watermark()
        self.remove_watermark()
        self.smooth_and_refine()
        self.save_output(save_path)
