// Array of code blocks for each question
const codeBlocks = [
    // Question 1: Basic Digital Image Operations
    `import cv2
import numpy as np
import matplotlib.pyplot as plt

def basic_operations():
    # Set image path
    image_path = 'input_image.jpg'
    
    # Read color image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Display image info
    print(f"Color image shape: {img.shape}")
    print(f"Color image type: {img.dtype}")
    print(f"Grayscale image shape: {gray.shape}")
    
    # Resize image to half
    height, width = img.shape[:2]
    resized = cv2.resize(img, (width//2, height//2))
    
    # Flip images
    horizontal_flip = cv2.flip(img, 1)  # 1 for horizontal flip
    vertical_flip = cv2.flip(img, 0)    # 0 for vertical flip
    
    # Extract region of interest (ROI)
    roi = img[100:200, 100:200]
    
    # Modify pixels
    modified = img.copy()
    modified[50:100, 50:100] = [255, 255, 255]  # White square
    modified[150:200, 150:200] = [0, 0, 0]      # Black square
    
    # Display images
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    
    plt.subplot(2, 3, 2)
    plt.imshow(gray, cmap='gray')
    plt.title('Grayscale')
    
    plt.subplot(2, 3, 3)
    plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    plt.title('Resized (50%)')
    
    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(horizontal_flip, cv2.COLOR_BGR2RGB))
    plt.title('Horizontal Flip')
    
    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    plt.title('ROI (100:200, 100:200)')
    
    plt.subplot(2, 3, 6)
    plt.imshow(cv2.cvtColor(modified, cv2.COLOR_BGR2RGB))
    plt.title('Modified Pixels')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    basic_operations()`,

    // Question 2: Image Transformations
    `import cv2
import numpy as np
import matplotlib.pyplot as plt

def image_transformations():
    # Set image path
    image_path = 'input_image.jpg'
    
    # Read and convert to grayscale
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Image Negative
    negative = 255 - gray
    
    # 2. Log Transformation
    # Formula: s = c * log(1 + r)
    c = 255 / np.log(1 + np.max(gray))
    log_image = c * np.log(1 + gray.astype(np.float32))
    log_image = np.uint8(log_image)
    
    # 3. Power Law (Gamma)
    # Formula: s = c * r^gamma
    gamma = 0.5
    gamma_image = np.uint8(255 * (gray / 255) ** gamma)
    
    # 4. Contrast Stretching
    min_val = np.min(gray)
    max_val = np.max(gray)
    stretched = np.uint8(255 * (gray - min_val) / (max_val - min_val))
    
    # 5. Bit Plane Slicing
    bit = 7  # Most significant bit
    bit_plane = np.uint8((gray >> bit) & 1) * 255
    
    # 6. Thresholding
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Display results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 3, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Original Grayscale')
    
    plt.subplot(2, 3, 2)
    plt.imshow(negative, cmap='gray')
    plt.title('Negative')
    
    plt.subplot(2, 3, 3)
    plt.imshow(log_image, cmap='gray')
    plt.title('Log Transformation')
    
    plt.subplot(2, 3, 4)
    plt.imshow(gamma_image, cmap='gray')
    plt.title(f'Gamma Correction (γ={gamma})')
    
    plt.subplot(2, 3, 5)
    plt.imshow(bit_plane, cmap='gray')
    plt.title(f'Bit Plane {bit}')
    
    plt.subplot(2, 3, 6)
    plt.imshow(threshold, cmap='gray')
    plt.title('Thresholding')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_transformations()`,

    // Question 3: Histogram Equalization
    `import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_equalization():
    # Set image path
    image_path = 'input_image.jpg'
    
    # Read and convert to grayscale
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    # Perform histogram equalization
    equalized = cv2.equalizeHist(gray)
    
    # Calculate equalized histogram
    hist_eq = cv2.calcHist([equalized], [0], None, [256], [0, 256])
    
    # Display results
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 2, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Original Grayscale')
    
    plt.subplot(2, 2, 2)
    plt.plot(hist)
    plt.title('Original Histogram')
    plt.xlim([0, 256])
    
    plt.subplot(2, 2, 3)
    plt.imshow(equalized, cmap='gray')
    plt.title('Equalized Image')
    
    plt.subplot(2, 2, 4)
    plt.plot(hist_eq)
    plt.title('Equalized Histogram')
    plt.xlim([0, 256])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    histogram_equalization()`,

    // Question 4: Spatial Domain Filtering
    `import cv2
import numpy as np
import matplotlib.pyplot as plt

def spatial_filtering():
    # Set image path
    image_path = 'input_image.jpg'
    
    # Read and convert to grayscale
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Averaging Filter (Smoothing)
    avg_kernel = np.ones((5, 5)) / 25  # 5x5 kernel with 1/25 value
    avg_filtered = cv2.filter2D(gray, -1, avg_kernel)
    
    # 2. Sharpening Filter
    sharp_kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ])
    sharp_filtered = cv2.filter2D(gray, -1, sharp_kernel)
    
    # 3. Unsharp Masking
    # Step 1: Blur the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Step 2: Create mask = original - blurred
    mask = gray - blurred
    # Step 3: Add mask to original
    unsharp = gray + mask
    unsharp = np.clip(unsharp, 0, 255).astype(np.uint8)
    
    # 4. Highboost Filtering
    k = 2.0  # Boost factor
    highboost = gray + k * mask
    highboost = np.clip(highboost, 0, 255).astype(np.uint8)
    
    # 5. Median Filtering
    median_filtered = cv2.medianBlur(gray, 5)
    
    # Display results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 3, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Original Grayscale')
    
    plt.subplot(2, 3, 2)
    plt.imshow(avg_filtered, cmap='gray')
    plt.title('Averaging Filter')
    
    plt.subplot(2, 3, 3)
    plt.imshow(sharp_filtered, cmap='gray')
    plt.title('Sharpening Filter')
    
    plt.subplot(2, 3, 4)
    plt.imshow(unsharp, cmap='gray')
    plt.title('Unsharp Masking')
    
    plt.subplot(2, 3, 5)
    plt.imshow(highboost, cmap='gray')
    plt.title('Highboost Filtering')
    
    plt.subplot(2, 3, 6)
    plt.imshow(median_filtered, cmap='gray')
    plt.title('Median Filtering')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    spatial_filtering()`,

    // Question 5: Frequency Domain Filtering
    `import cv2
import numpy as np
import matplotlib.pyplot as plt

def frequency_filtering():
    # Set image path
    image_path = 'input_image.jpg'
    
    # Read and convert to grayscale
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Expand image to optimal size for FFT
    rows, cols = gray.shape
    padded = cv2.copyMakeBorder(gray, 0, rows, 0, cols, cv2.BORDER_CONSTANT)
    
    # Step 1: Compute DFT
    dft = cv2.dft(np.float32(padded), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    # Calculate magnitude spectrum for display
    magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    magnitude_display = 20 * np.log(magnitude + 1)  # Add 1 to avoid log(0)
    
    # Create center coordinates
    rows, cols = padded.shape
    center_row, center_col = rows // 2, cols // 2
    
    # Create distance matrix from center
    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((x - center_col)**2 + (y - center_row)**2)
    
    # Create filters
    radius = 50  # Filter radius
    
    # Ideal Lowpass Filter
    ideal_mask = np.ones((rows, cols, 2), dtype=np.float32)
    ideal_mask[distance > radius] = 0
    
    # Butterworth Lowpass Filter
    n = 2  # Order
    butterworth_mask = np.ones((rows, cols, 2), dtype=np.float32)
    butterworth_mask = 1 / (1 + (distance / radius)**(2*n))
    butterworth_mask = np.stack([butterworth_mask, butterworth_mask], axis=2)
    
    # Gaussian Lowpass Filter
    gaussian_mask = np.exp(-(distance**2) / (2 * (radius**2)))
    gaussian_mask = np.stack([gaussian_mask, gaussian_mask], axis=2)
    
    # Apply filters
    ideal_filtered_dft = dft_shift * ideal_mask
    butterworth_filtered_dft = dft_shift * butterworth_mask
    gaussian_filtered_dft = dft_shift * gaussian_mask
    
    # Inverse DFT
    # Ideal
    ideal_filtered = cv2.idft(np.fft.ifftshift(ideal_filtered_dft))
    ideal_filtered = cv2.magnitude(ideal_filtered[:, :, 0], ideal_filtered[:, :, 1])
    ideal_filtered = np.clip(ideal_filtered, 0, 255).astype(np.uint8)
    
    # Butterworth
    butterworth_filtered = cv2.idft(np.fft.ifftshift(butterworth_filtered_dft))
    butterworth_filtered = cv2.magnitude(butterworth_filtered[:, :, 0], butterworth_filtered[:, :, 1])
    butterworth_filtered = np.clip(butterworth_filtered, 0, 255).astype(np.uint8)
    
    # Gaussian
    gaussian_filtered = cv2.idft(np.fft.ifftshift(gaussian_filtered_dft))
    gaussian_filtered = cv2.magnitude(gaussian_filtered[:, :, 0], gaussian_filtered[:, :, 1])
    gaussian_filtered = np.clip(gaussian_filtered, 0, 255).astype(np.uint8)
    
    # Display results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 3, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Original Image')
    
    plt.subplot(2, 3, 2)
    plt.imshow(np.log1p(magnitude_display), cmap='gray')
    plt.title('Frequency Spectrum')
    
    plt.subplot(2, 3, 3)
    plt.imshow(ideal_filtered[:rows//2, :cols//2], cmap='gray')
    plt.title('Ideal Lowpass Filter')
    
    plt.subplot(2, 3, 4)
    plt.imshow(butterworth_filtered[:rows//2, :cols//2], cmap='gray')
    plt.title('Butterworth Lowpass Filter')
    
    plt.subplot(2, 3, 5)
    plt.imshow(gaussian_filtered[:rows//2, :cols//2], cmap='gray')
    plt.title('Gaussian Lowpass Filter')
    
    plt.tight_layout()
    plt.show()
    
    # Print filter equations
    print("Filter Equations:")
    print("Ideal Lowpass Filter: H(u,v) = 1 if D(u,v) ≤ D₀, 0 otherwise")
    print("Butterworth Lowpass Filter: H(u,v) = 1 / (1 + [D(u,v)/D₀]²ⁿ)")
    print("Gaussian Lowpass Filter: H(u,v) = e^(-D(u,v)²/2D₀²)")

if __name__ == "__main__":
    frequency_filtering()`,

    // Question 6: Morphological Operations
    `import cv2
import numpy as np
import matplotlib.pyplot as plt

def morphological_operations():
    # Set image path
    image_path = 'input_image.jpg'
    
    # Read and convert to grayscale
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Convert to binary image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Create kernel (structuring element)
    kernel = np.ones((5, 5), np.uint8)
    
    # Apply morphological operations
    # 1. Dilation
    dilation = cv2.dilate(binary, kernel, iterations=1)
    
    # 2. Erosion
    erosion = cv2.erode(binary, kernel, iterations=1)
    
    # 3. Opening (Erosion followed by Dilation)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # 4. Closing (Dilation followed by Erosion)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 5. Boundary Extraction
    boundary = binary - erosion
    
    # 6. Hit-or-Miss Transform
    kernel1 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    kernel2 = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], np.uint8)
    
    eroded = cv2.erode(binary, kernel1)
    eroded_complement = cv2.erode(cv2.bitwise_not(binary), kernel2)
    hit_miss = cv2.bitwise_and(eroded, eroded_complement)
    
    # Display results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 4, 1)
    plt.imshow(binary, cmap='gray')
    plt.title('Binary Image')
    
    plt.subplot(2, 4, 2)
    plt.imshow(dilation, cmap='gray')
    plt.title('Dilation')
    
    plt.subplot(2, 4, 3)
    plt.imshow(erosion, cmap='gray')
    plt.title('Erosion')
    
    plt.subplot(2, 4, 4)
    plt.imshow(opening, cmap='gray')
    plt.title('Opening')
    
    plt.subplot(2, 4, 5)
    plt.imshow(closing, cmap='gray')
    plt.title('Closing')
    
    plt.subplot(2, 4, 6)
    plt.imshow(boundary, cmap='gray')
    plt.title('Boundary')
    
    plt.subplot(2, 4, 7)
    plt.imshow(hit_miss, cmap='gray')
    plt.title('Hit-or-Miss')
    
    plt.tight_layout()
    plt.show()
    
    # Print operations in simple form
    print("Morphological Operations:")
    print("Dilation: Expands the boundaries of foreground objects")
    print("Erosion: Shrinks the boundaries of foreground objects")
    print("Opening: Erosion followed by dilation (removes small objects)")
    print("Closing: Dilation followed by erosion (fills small holes)")
    print("Boundary: Original - Eroded image")
    print("Hit-or-Miss: Finds specific patterns in binary images")

if __name__ == "__main__":
    morphological_operations()`,

    // Question 7: Edge Detection
    `import cv2
import numpy as np
import matplotlib.pyplot as plt

def edge_detection():
    # Set image path
    image_path = 'input_image.jpg'
    
    # Read and convert to grayscale
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Define Prewitt kernels
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    
    # Define Sobel kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    # Apply kernels
    prewitt_gx = cv2.filter2D(gray, -1, prewitt_x)
    prewitt_gy = cv2.filter2D(gray, -1, prewitt_y)
    
    sobel_gx = cv2.filter2D(gray, -1, sobel_x)
    sobel_gy = cv2.filter2D(gray, -1, sobel_y)
    
    # Calculate magnitude
    prewitt_mag = np.sqrt(prewitt_gx**2 + prewitt_gy**2).astype(np.uint8)
    sobel_mag = np.sqrt(sobel_gx**2 + sobel_gy**2).astype(np.uint8)
    
    # Display results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 4, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Original Grayscale')
    
    plt.subplot(2, 4, 2)
    plt.imshow(prewitt_gx, cmap='gray')
    plt.title('Prewitt Horizontal')
    
    plt.subplot(2, 4, 3)
    plt.imshow(prewitt_gy, cmap='gray')
    plt.title('Prewitt Vertical')
    
    plt.subplot(2, 4, 4)
    plt.imshow(prewitt_mag, cmap='gray')
    plt.title('Prewitt Magnitude')
    
    plt.subplot(2, 4, 5)
    plt.imshow(gray, cmap='gray')
    plt.title('Original Grayscale')
    
    plt.subplot(2, 4, 6)
    plt.imshow(sobel_gx, cmap='gray')
    plt.title('Sobel Horizontal')
    
    plt.subplot(2, 4, 7)
    plt.imshow(sobel_gy, cmap='gray')
    plt.title('Sobel Vertical')
    
    plt.subplot(2, 4, 8)
    plt.imshow(sobel_mag, cmap='gray')
    plt.title('Sobel Magnitude')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    edge_detection()`,

    // Question 8: Global Thresholding
    `import cv2
import numpy as np
import matplotlib.pyplot as plt

def global_thresholding():
    # Set image path
    image_path = 'input_image.jpg'
    
    # Read and convert to grayscale
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    # Simple global thresholding
    threshold_value = 127
    _, binary1 = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Otsu's method (automatic thresholding)
    otsu_thresh, binary2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Display results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Original Grayscale')
    
    plt.subplot(2, 2, 2)
    plt.plot(hist)
    plt.title('Histogram')
    plt.xlim([0, 256])
    
    plt.subplot(2, 2, 3)
    plt.imshow(binary1, cmap='gray')
    plt.title(f'Global Threshold (T={threshold_value})')
    
    plt.subplot(2, 2, 4)
    plt.imshow(binary2, cmap='gray')
    plt.title(f'Otsu Threshold (T={otsu_thresh})')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    global_thresholding()`,

    // Question 9: Perceptron Implementation
    `import numpy as np
import matplotlib.pyplot as plt

def perceptron_implementation():
    # Simple Perceptron class
    class SimplePerceptron:
        def __init__(self, learning_rate=0.1, iterations=100):
            self.learning_rate = learning_rate
            self.iterations = iterations
            self.weights = None
            self.bias = None
        
        def train(self, X, y):
            # Initialize weights and bias
            input_size = X.shape[1]
            self.weights = np.zeros(input_size)
            self.bias = 0
            
            # Training loop
            for _ in range(self.iterations):
                for i in range(len(X)):
                    # Calculate prediction
                    x = X[i]
                    prediction = self.predict_single(x)
                    
                    # Update weights and bias
                    error = y[i] - prediction
                    self.weights += self.learning_rate * error * x
                    self.bias += self.learning_rate * error
        
        def predict_single(self, x):
            # Calculate weighted sum
            z = np.dot(x, self.weights) + self.bias
            # Apply step function
            return 1 if z >= 0 else 0
        
        def predict(self, X):
            return [self.predict_single(x) for x in X]
    
    # Input data for logic gates
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    
    # AND gate output
    y_and = np.array([0, 0, 0, 1])
    
    # OR gate output
    y_or = np.array([0, 1, 1, 1])
    
    # Train perceptrons
    and_perceptron = SimplePerceptron(learning_rate=0.1, iterations=50)
    and_perceptron.train(X, y_and)
    
    or_perceptron = SimplePerceptron(learning_rate=0.1, iterations=50)
    or_perceptron.train(X, y_or)
    
    # Make predictions
    and_predictions = and_perceptron.predict(X)
    or_predictions = or_perceptron.predict(X)
    
    # Print results
    print("AND Gate Perceptron:")
    print(f"Weights: {and_perceptron.weights}, Bias: {and_perceptron.bias}")
    print("Truth Table:")
    for i in range(len(X)):
        print(f"{X[i][0]} AND {X[i][1]} = {and_predictions[i]}")
    
    print("\nOR Gate Perceptron:")
    print(f"Weights: {or_perceptron.weights}, Bias: {or_perceptron.bias}")
    print("Truth Table:")
    for i in range(len(X)):
        print(f"{X[i][0]} OR {X[i][1]} = {or_predictions[i]}")
    
    # Display decision boundaries
    plt.figure(figsize=(12, 5))
    
    # AND Gate
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_and, cmap=plt.cm.coolwarm, s=100)
    
    # Decision boundary line: w1*x1 + w2*x2 + b = 0
    if and_perceptron.weights[1] != 0:
        x1 = np.linspace(-0.5, 1.5, 100)
        x2 = -(and_perceptron.weights[0] * x1 + and_perceptron.bias) / and_perceptron.weights[1]
        plt.plot(x1, x2, 'k-')
    
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.grid(True)
    plt.title('AND Gate')
    
    # OR Gate
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=y_or, cmap=plt.cm.coolwarm, s=100)
    
    # Decision boundary line
    if or_perceptron.weights[1] != 0:
        x1 = np.linspace(-0.5, 1.5, 100)
        x2 = -(or_perceptron.weights[0] * x1 + or_perceptron.bias) / or_perceptron.weights[1]
        plt.plot(x1, x2, 'k-')
    
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.grid(True)
    plt.title('OR Gate')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    perceptron_implementation()`
];

// Question titles
const questionTitles = [
    "Question 1: Basic Digital Image Operations",
    "Question 2: Image Transformations",
    "Question 3: Histogram Equalization",
    "Question 4: Spatial Domain Filtering",
    "Question 5: Frequency Domain Filtering",
    "Question 6: Morphological Operations",
    "Question 7: Edge Detection",
    "Question 8: Global Thresholding",
    "Question 9: Perceptron Implementation"
];

// Show the selected question
function showQuestion(index) {
    // Update active button
    const buttons = document.querySelectorAll('.question-btn');
    buttons.forEach((btn, i) => {
        if (i === index) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });

    // Update title and code
    document.getElementById('question-title').textContent = questionTitles[index];
    document.getElementById('code-display').textContent = codeBlocks[index];
}

// Copy code function
function copyCode() {
    const codeText = document.getElementById('code-display').textContent;
    
    // Create a temporary textarea to copy from
    const textarea = document.createElement('textarea');
    textarea.value = codeText;
    document.body.appendChild(textarea);
    textarea.select();
    
    // Execute copy command
    document.execCommand('copy');
    
    // Clean up
    document.body.removeChild(textarea);
    
    // Show toast notification
    const toast = document.getElementById('toast');
    toast.className = 'show';
    
    // Hide toast after 3 seconds
    setTimeout(() => {
        toast.className = toast.className.replace('show', '');
    }, 3000);
}

// Initialize the first question on page load
document.addEventListener('DOMContentLoaded', () => {
    showQuestion(0);
}); 