import cv2
import os
import numpy as np
import matplotlib.pyplot as plt



class BackgroundRemover:
    
    def __init__(self, input_video_path, output_video_path, custom_bg_path,
                 threshold=30, box_kernel=(5, 5), median_kernel=5,
                 dynamic_bg=False, alpha=0.005):
        """
        Initializes the BackgroundRemover with required parameters.
        
        :param input_video_path: Path to the input video.
        :param output_video_path: Path where the output video will be saved.
        :param custom_bg_path: Path to the custom background image (grayscale).
        :param threshold: Threshold value for binary mask creation.
        :param box_kernel: Kernel size for box filter.
        :param median_kernel: Kernel size for median filter.
        :param dynamic_bg: Boolean flag; if True, use a running average for a dynamic background.
        :param alpha: Learning rate for running average (only used if dynamic_bg is True).
        """
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        self.custom_bg_path = custom_bg_path
        self.threshold = threshold
        self.box_kernel = box_kernel
        self.median_kernel = median_kernel
        self.dynamic_bg = dynamic_bg
        self.alpha = alpha  # used for updating the dynamic background model
        
        self.frames = []             # To store all video frames
        self.processed_frames = []   # To store processed frames
        self.fps = None              # Frame rate of the input video
        self.frame_size = None       # (width, height) of the video frames
        self.bg_model = None         # Background model (reference frame or running average)
        self.custom_bg = None        # Custom background image (grayscale)

    # def read_image(self):
    #     """
    #     Reads the image in grayscale and displays it.
    #     Returns:
    #         image (numpy.ndarray): The grayscale image.
    #     """
    #     image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

    #     if image is None:
    #         raise FileNotFoundError(f"Image file '{self.image_path}' not found.")

    #     self.display_image(image, "Original Image")

    #     return image

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

    def extract_frames(self):
        """Extracts and converts all video frames to grayscale."""
        cap = cv2.VideoCapture(self.input_video_path)
        
        if not cap.isOpened():
            raise IOError("Error opening video file.")
        
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        ret, frame = cap.read()

        self.h, self.w = frame.shape[:2]

        # print(f"Height: {self.h}, Width: {self.w}") # For, testing
        
        while ret:
            # Converting to grayscale if the frame is not already grayscale
            if len(frame.shape) == 3:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray_frame = frame
            
            # Setting frame size (width, height) from the first frame
            if self.frame_size is None:
                self.frame_size = (gray_frame.shape[1], gray_frame.shape[0])
            
            self.frames.append(gray_frame)
            ret, frame = cap.read()

        cap.release()
        
        print(f"Extracted {len(self.frames)} frames from video.")

        self.custom_background_loader(self.h, self.w)

        print("\nCustom background image loaded.")

    def custom_background_loader(self, height, width):
        # # Creating a grayscale background image with a gradient
        # background = np.zeros((height, width), dtype=np.uint8)

        # # Filling the image with a gradient
        # for i in range(height):
            
        #     for j in range(width):
        #         background[i, j] = int((i + j) / (height + width) * 255)

        # Creating a uniform mid-gray background (e.g., 127)
        background = np.full((height, width), 0, dtype=np.uint8)

        # # Displaying the image
        # cv2.imshow('Grayscale Background', background)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Defining the path to save the image
        save_path = self.custom_bg_path

        # Checking if the image already exists and delete it if it does
        if os.path.exists(save_path):
            os.remove(save_path)

        # Saving the new image as a .jpg file
        cv2.imwrite(save_path, background)
    
    def initialize_background_model(self):
        """
        Loads and resizes the custom background image.
        Initializes the background model:
          - For static background, use the first frame.
          - For dynamic background, initialize a running average.
        """
        # Loading custom background image in grayscale and resize it to the frame size
        self.custom_bg = cv2.imread(self.custom_bg_path, cv2.IMREAD_GRAYSCALE)
        
        if self.custom_bg is None:
            raise IOError("Custom background image not found.")
        
        self.custom_bg = cv2.resize(self.custom_bg, self.frame_size)
        
        # Initializing background model
        if not self.dynamic_bg:
            # Static background: use the first frame as reference
            self.bg_model = self.frames[0]
        else:
            # Dynamic background: use the first frame as the initial model in float32 format
            self.bg_model = self.frames[0].astype(np.float32)

    def gamma_correct(self, image, gamma=1.2):
        # Build a lookup table first
        invGamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** invGamma * 255 
                        for i in np.arange(256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    def process_frames(self):
        """Processes each frame: background detection, mask refinement, and background replacement."""
        print()

        for i, frame in enumerate(self.frames):
            # self.display_image(frame, f"Frame {i+1}")

            # Updating background model dynamically if required
            if self.dynamic_bg:
                # Accumulating weighted average (running average)
                cv2.accumulateWeighted(frame, self.bg_model, self.alpha)
                bg_frame = cv2.convertScaleAbs(self.bg_model)
            else:
                bg_frame = self.bg_model

            # self.display_image(bg_frame, "Background Model")
            
            # Computing the difference between the current frame and the background model.
            # Using cv2.subtract helps avoid negative values, but if you use direct subtraction,
            # you might need to normalize negative values later.
            # diff = cv2.subtract(frame, bg_frame)
            corrected_frame = self.gamma_correct(frame, gamma=1.2)
            diff = cv2.absdiff(frame, bg_frame) # absdiff takes the absolute difference of the two images, helping capture both lighter and darker variations that occur when the subject moves through shadows.

            # self.display_image(diff, "Difference Image")
            
            # Threshold the difference to create a binary mask (foreground vs. background)
            # _, mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
            # If 30 is missing parts of the subject, try lowering to 20, 15, or even 10
            _, mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

            # self.display_image(mask, "Binary Mask")
            
            # Refining the mask:
            # Smoothing the mask using a box (averaging) filter.
            mask_smooth = cv2.blur(mask, self.box_kernel)
            # Further reducing noise with a median filter.
            mask_refined = cv2.medianBlur(mask_smooth, self.median_kernel)
            
            # To clearly distinguish the foreground, threshold the refined mask.
            # _, final_mask = cv2.threshold(mask_refined, 127, 255, cv2.THRESH_BINARY)
            # Instead of thresholding at 127 again:
            final_mask = mask_refined
            # Or, if needed, adjust the threshold value based on your observations.
            
            # Creating an inverse mask for background replacement.
            bg_mask = cv2.bitwise_not(final_mask)
            
            # Copying the current frame for background replacement.
            processed_frame = frame.copy()
            # Replacing background pixels (where bg_mask is 255) with the custom background.
            processed_frame[bg_mask == 255] = self.custom_bg[bg_mask == 255]
            
            # Handling any negative pixel values via normalization.
            processed_frame = self.normalize_frame(processed_frame)
            
            self.processed_frames.append(processed_frame)

            print(f"Processed frame {i+1} of {len(self.frames)}")
    
    def normalize_frame(self, frame):
        """
        Normalizes the frame so that all pixel values fall between 0 and 255.
        This method checks for negative values and applies a shift-and-scale if necessary.
        """
        min_val = np.min(frame)

        if min_val < 0:
            # Shifting values so that the minimum becomes 0
            frame = frame - min_val
            # Scaling the image so the maximum is 255
            frame = (frame / np.max(frame)) * 255
            frame = frame.astype(np.uint8)

        return frame
    
    def reconstruct_video(self):
        """Recombines the processed frames into a new video with the same fps and frame size."""
        # Defining the codec and create VideoWriter object.
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(self.output_video_path, fourcc, self.fps, self.frame_size, isColor=False)
        
        for i, frame in enumerate(self.processed_frames):
            out.write(frame)

        out.release()
        
        print(f"\nVideo reconstruction complete. Output saved to {self.output_video_path}")
    
    def run(self):
        """Executes the full pipeline: frame extraction, background modeling, processing, and reconstruction."""
        self.extract_frames()
        self.initialize_background_model()
        self.process_frames()
        self.reconstruct_video()

class BackgroundRemover1:
    
    def __init__(self, input_video_path, output_video_path, custom_bg_path,
                 threshold=30, box_kernel=(5, 5), median_kernel=5,
                 dynamic_bg=True, alpha=0.001):
        """
        Initializes the BackgroundRemover with required parameters.
        
        :param input_video_path: Path to the input video.
        :param output_video_path: Path where the output video will be saved.
        :param custom_bg_path: Path to the custom background image (grayscale).
        :param threshold: Threshold value for binary mask creation.
        :param box_kernel: Kernel size for box filter.
        :param median_kernel: Kernel size for median filter.
        :param dynamic_bg: Boolean flag; if True, use a running average for a dynamic background.
        :param alpha: Learning rate for running average (only used if dynamic_bg is True).
        """
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        self.custom_bg_path = custom_bg_path
        self.threshold = threshold
        self.box_kernel = box_kernel
        self.median_kernel = median_kernel
        self.dynamic_bg = dynamic_bg
        self.alpha = alpha  # used for updating the dynamic background model
        
        self.frames = []             # To store all video frames
        self.processed_frames = []   # To store processed frames
        self.fps = None              # Frame rate of the input video
        self.frame_size = None       # (width, height) of the video frames
        self.bg_model = None         # Background model (reference frame or running average)
        self.custom_bg = None        # Custom background image (grayscale)

    def display_image(self, image, title="Image"):
        """
        Displays an image using matplotlib.
        Args:
            image (numpy.ndarray): Image to display.
            title (str): Title for the displayed image.
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(6,6))
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()

    def extract_frames(self):
        """Extracts and converts all video frames to grayscale."""
        import cv2
        
        cap = cv2.VideoCapture(self.input_video_path)
        
        if not cap.isOpened():
            raise IOError("Error opening video file.")
        
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        
        ret, frame = cap.read()

        self.h, self.w = frame.shape[:2]
        
        while ret:
            # Converting to grayscale if the frame is not already grayscale
            if len(frame.shape) == 3:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray_frame = frame
            
            # Setting frame size (width, height) from the first frame
            if self.frame_size is None:
                self.frame_size = (gray_frame.shape[1], gray_frame.shape[0])
            
            self.frames.append(gray_frame)
            ret, frame = cap.read()

        cap.release()
        
        print(f"Extracted {len(self.frames)} frames from video.")

        self.custom_background_loader(self.h, self.w)
        
        print("\nCustom background image loaded.")

    def custom_background_loader(self, height, width):
        """
        Creates a custom uniform background image and saves it.
        """
        import cv2, os, numpy as np
        
        # Creating a uniform black background (all zeros). Adjust value if needed.
        background = np.full((height, width), 0, dtype=np.uint8)

        save_path = self.custom_bg_path
        
        if os.path.exists(save_path):
            os.remove(save_path)
        
        cv2.imwrite(save_path, background)
    
    def initialize_background_model(self):
        """
        Loads and resizes the custom background image.
        Initializes the background model using a robust approach:
          - Instead of using just the first frame, use the median of the first N frames.
          - For static background, this median is used as the reference.
          - For dynamic background, this serves as the initial model (in float32).
        """
        import cv2, numpy as np
        
        # Loading and resizing custom background image
        self.custom_bg = cv2.imread(self.custom_bg_path, cv2.IMREAD_GRAYSCALE)
        
        if self.custom_bg is None:
            raise IOError("Custom background image not found.")
        
        self.custom_bg = cv2.resize(self.custom_bg, self.frame_size)
        
        # Robust initialization: use median of first N frames
        N = min(10, len(self.frames))
        
        initial_frames = self.frames[:N]
        
        self.bg_model = np.median(np.stack(initial_frames), axis=0).astype(np.float32)
        # For static mode, we keep the bg_model fixed.
        # For dynamic mode, we'll update bg_model selectively in process_frames.

    def gamma_correct(self, image, gamma=1.2):
        """
        Applies gamma correction to adjust brightness.
        """
        import cv2, numpy as np
        
        invGamma = 1.0 / gamma
        
        table = np.array([(i / 255.0) ** invGamma * 255 
                        for i in np.arange(256)]).astype("uint8")
        
        return cv2.LUT(image, table)
    
    def process_frames(self):
        """Processes each frame: background subtraction, mask refinement, and background replacement."""
        import cv2, numpy as np
        
        print()
        for i, frame in enumerate(self.frames):
            # Applying gamma correction to current frame for consistency
            corrected_frame = self.gamma_correct(frame, gamma=1.2)
            
            # Getting the current background frame from the model and apply gamma correction
            bg_frame = cv2.convertScaleAbs(self.bg_model)
            corrected_bg = self.gamma_correct(bg_frame, gamma=1.2)
            
            # Computing the absolute difference between the gamma-corrected frame and background
            diff = cv2.absdiff(corrected_frame, corrected_bg)
            
            # Using the provided threshold (adaptive thresholding can be implemented as needed)
            # _, mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)

            # Defining parameters for adaptive thresholding
            block_size = 11  # Must be an odd number (e.g., 11, 15, 21) and greater than 1
            C = 2            # Constant subtracted from the computed mean or weighted sum

            # Applying adaptive thresholding on the difference image 'diff'
            mask = cv2.adaptiveThreshold(diff, 
                                        255, 
                                        # cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 
                                        block_size, 
                                        C)
            
            # Mask refinement: smoothing with box filter, median filter, then morphological operations
            mask_smooth = cv2.blur(mask, self.box_kernel)
            mask_refined = cv2.medianBlur(mask_smooth, self.median_kernel)
            
            # # Morphological opening and closing to remove small artifacts
            # kernel = np.ones((3, 3), np.uint8)
            # refined_mask = cv2.morphologyEx(mask_refined, cv2.MORPH_OPEN, kernel)
            # refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
            
            # Optional: apply a slight Gaussian blur to soften edges for smoother background replacement
            # final_mask = cv2.GaussianBlur(refined_mask, (3, 3), 0)
            final_mask = cv2.GaussianBlur(mask_refined, (3, 3), 0)
            
            # Creating an inverse mask for background replacement
            bg_mask = cv2.bitwise_not(final_mask)
            
            # Replacing background: copy frame then replace pixels where bg_mask is 255 with custom background pixels
            processed_frame = frame.copy()
            processed_frame[bg_mask == 255] = self.custom_bg[bg_mask == 255]
            
            # Normalizing processed frame to ensure pixel values are within 0-255
            processed_frame = self.normalize_frame(processed_frame)
            self.processed_frames.append(processed_frame)
            
            # If dynamic background mode is enabled, update the background model selectively:
            # Only update background pixels (where final_mask indicates background)
            if self.dynamic_bg:
                background_pixels = cv2.bitwise_not(final_mask)
                cv2.accumulateWeighted(frame, self.bg_model, self.alpha, mask=background_pixels)
            
            print(f"Processed frame {i+1} of {len(self.frames)}")
    
    def normalize_frame(self, frame):
        """
        Normalizes the frame so that all pixel values fall between 0 and 255.
        """
        import numpy as np
        
        min_val = np.min(frame)
        
        if min_val < 0:
            frame = frame - min_val
            frame = (frame / np.max(frame)) * 255
            frame = frame.astype(np.uint8)
        
        return frame
    
    def reconstruct_video(self):
        """Recombines the processed frames into a new video with the same fps and frame size."""
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
        out = cv2.VideoWriter(self.output_video_path, fourcc, self.fps, self.frame_size, isColor=False)
        
        for frame in self.processed_frames:
            out.write(frame)
        
        out.release()
        
        print(f"\nVideo reconstruction complete. Output saved to {self.output_video_path}")
    
    def run(self):
        """Executes the full pipeline: frame extraction, background modeling, processing, and reconstruction."""
        self.extract_frames()
        self.initialize_background_model()
        self.process_frames()
        self.reconstruct_video()
