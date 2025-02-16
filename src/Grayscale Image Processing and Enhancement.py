import cv2
import numpy as np
import matplotlib.pyplot as plt
import os



class ImageProcessor:

    def __init__(self, image_path):
        """
        Initialize the ImageProcessor with the path to the grayscale image.
        Reads and displays the original image.
        """
        self.image_path = image_path
        self.original = self.read_image()

        filename = os.path.basename(self.image_path)

        self.image_name, ext = os.path.splitext(filename)

        # print(self.image_name)  # For, testing

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

    def reduce_noise(self, kernel_size=3, method="median"):  # Or even 9 for stronger noise reduction
        """
        Reduces high-frequency noise using a median filter.
        Args:
            kernel_size (int): Size of the median filter kernel (must be an odd integer).
        Returns:
            median_filtered (numpy.ndarray): The noise-reduced image.
        """
        if method == "median":
            self.noise_filtered = cv2.medianBlur(self.original, kernel_size)
        if method == "gaussian":
            self.noise_filtered = cv2.GaussianBlur(self.original, (kernel_size, kernel_size), 1.5)
        if method == "bilateral": # This can remove noise while keeping digit edges clearer.
            self.noise_filtered = cv2.bilateralFilter(self.original, 9, 75, 75)

        # In many OCR or digit-recognition pipelines, a slight blur can actually improve thresholding results because it helps unify strokes. The key is to not blur so aggressively that digits lose shape.

        self.display_image(self.noise_filtered, "After Noise Reduction")

        return self.noise_filtered

    def enhance_contrast(self, method="gamma", gamma=2.0):
        """
        Enhances contrast using either Log or Gamma (Power-law) transformation.
        Args:
            method (str): 'log' for Log transformation, 'gamma' for Power-law transformation.
            gamma (float): Gamma value for Power-law transformation (default=1.5).
        Returns:
            contrast_enhanced (numpy.ndarray): The contrast-enhanced image.
        """
        if not hasattr(self, 'noise_filtered'):
            raise ValueError("Please apply noise reduction before contrast enhancement.")

        if method == "log":
            # Log transformation is less effective for images with uneven lighting.
            c = 255 / np.log(1 + np.max(self.noise_filtered))
            contrast_enhanced = c * np.log(1 + self.noise_filtered.astype(np.float64))
            contrast_enhanced = np.array(contrast_enhanced, dtype=np.uint8)
        elif method == "gamma":
            # # Gamma correction allows more flexible enhancement.
            # If digits look too dark after enhancement, increase gamma (e.g., gamma=2.0).
            # If digits become too bright, lower gamma (e.g., gamma=1.2).
            contrast_enhanced = np.power(self.noise_filtered / 255.0, gamma) * 255
            contrast_enhanced = contrast_enhanced.astype(np.uint8)
        else:
            raise ValueError("Invalid contrast enhancement method. Choose 'log' or 'gamma'.")

        self.contrast_enhanced = contrast_enhanced
        self.display_image(self.contrast_enhanced, f"Contrast Enhancement ({method.capitalize()})")

        return self.contrast_enhanced

    def laplacian_sharpen(self):
        """
        Sharpens the contrast-enhanced image using a Laplacian filter.
        Returns:
            sharpened (numpy.ndarray): The sharpened image.
        """
        if not hasattr(self, 'contrast_enhanced'):
            raise ValueError("Please enhance contrast before applying Laplacian sharpening.")
        
        laplacian = cv2.Laplacian(self.contrast_enhanced, cv2.CV_64F, ksize=3)

        # Converting laplacian to uint8
        laplacian = np.uint8(np.absolute(laplacian))
        
        # Weighted combination: alpha = strength of original image, beta = strength of edges
        alpha, beta = 1.5, -0.5
        sharpened = cv2.addWeighted(self.contrast_enhanced, alpha, -laplacian, beta, 0)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        self.sharpened = sharpened
        
        self.display_image(self.sharpened, "After Laplacian Sharpening")
        
        return self.sharpened

    def binarize(self, method="global", threshold=127):
        """
        Converts the sharpened image into a binary (black-and-white) image.
        Args:
            method (str): 'global' for global thresholding or 'adaptive' for adaptive thresholding.
            threshold (int): Threshold value for global thresholding.
        Returns:
            binary_image (numpy.ndarray): The binarized image.
        """
        if not hasattr(self, 'sharpened'):
            raise ValueError("Please sharpen the image before binarization.")
        
        if method == "global":
            ret, self.binary_image = cv2.threshold(
                self.sharpened, threshold, 255, cv2.THRESH_BINARY_INV)
        
            self.display_image(self.binary_image, "Final Binary Image (Global Thresholding)")
        elif method == "adaptive":
            # Increase block size if digits vary in local contrast
            block_size = 15
            C = 2  
            self.binary_image = cv2.adaptiveThreshold(
                self.sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, block_size, C
            )
        
            self.display_image(self.binary_image, "Final Binary Image (Adaptive Thresholding)")
        else:
            raise ValueError("Invalid method specified. Choose 'global' or 'adaptive'.")
    
        # Morphological Opening or Closing (optional)
        kernel = np.ones((3,3), np.uint8)
        
        # morphological opening to remove noise (if you have both noise specks)
        self.binary_image = cv2.morphologyEx(self.binary_image, cv2.MORPH_OPEN, kernel)
        self.display_image(self.binary_image, "After Morphological Opening")

        # # (If digits are broken, you might try morphological closing or a combination)
        # self.binary_image = cv2.morphologyEx(self.binary_image, cv2.MORPH_CLOSE, kernel)
        # self.display_image(self.binary_image, "After Morphological Closing")

        # # Summary:
        # Even with the best denoising, your final binarization can make or break digit clarity. If you see fragmented digits or leftover noise:

        # Adaptive Thresholding: Increase or decrease the block_size or C to better isolate digits.
        # Morphological Opening: Removes leftover small specks after thresholding.
        # Morphological Closing: Reconnects broken parts of digits
        
        return self.binary_image

    def save_image(self, save_path):
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
        full_save_path = os.path.join(results_dir, f"{save_path}_{self.image_name}_processed.png")
        
        # Removing existing file if it exists
        if os.path.exists(full_save_path):
            os.remove(full_save_path)

        if not hasattr(self, 'binary_image'):
            raise ValueError("Binarization has not been performed. Nothing to save.")
        
        # Saving the image
        cv2.imwrite(full_save_path, self.binary_image)
        
        print(f"Image saved at {full_save_path}")

    def process_pipeline(self, save_path):
        """
        Runs the complete image processing pipeline and saves the final image.
        Args:
            save_path (str): Path where the final image will be saved.
            threshold (int): Threshold value for global thresholding.
            method (str): 'global' or 'adaptive' thresholding method.
        """
        ## Sample hyperparameters:
        # noise_method="median", 
        # noise_kernel=7, 
        # contrast_method="gamma", 
        # gamma=2.0,
        # sharpen_alpha=1.5, 
        # sharpen_beta=-0.5,
        # bin_method="adaptive",
        # block_size=15, 
        # C=2

        self.reduce_noise()
        self.enhance_contrast()
        self.laplacian_sharpen()
        self.binarize()
        self.save_image(save_path)