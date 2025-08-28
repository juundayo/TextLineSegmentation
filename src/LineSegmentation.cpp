#include "LineSegmentation.hpp"
#include <filesystem>

namespace fs = std::filesystem;

LineSegmentation::LineSegmentation(string path_of_image, string out) {
    this->image_path = path_of_image; // Path to input image.
    this->OUT_PATH   = out;           // Output directory path.

    // Initializing Sieve for prime number calculations.
    // (used in probability calculations)
    sieve();

    // Clearing containers to ensure clean state.
    primes.clear();         // Prime numbers for factorization.
    chunks.clear();         // Vertical image chunks.
    chunk_map.clear();      // Mapping of chunks.
    map_valley.clear();     // Valley points mapping.
    initial_lines.clear();  // Initially detected lines.
    outFinal_lines.clear(); // Final refined lines.
    line_regions.clear();   // Regions between lines.
    contours.clear();       // Character/component contours.
}

// Reads the input image in both color and grayscale formats.
void LineSegmentation::read_image() {
    this->color_img = cv::imread(this->image_path, cv::IMREAD_COLOR);
    this->grey_img = cv::imread(this->image_path, cv::IMREAD_GRAYSCALE);
}

// Image preprocessing with noise reduction and binarization.
void LineSegmentation::pre_process_image() {
    cv::Mat preprocessed_img, smoothed_img;

    // Noise reduction using a simple 3x3 blur filter.
    cv::blur(grey_img, smoothed_img, Size(3, 3), Point(-1, -1));

    // OTSU threshold and binarization.
    cv::threshold(smoothed_img, binary_img, 0.0, 255,
                  cv::THRESH_BINARY | cv::THRESH_OTSU);
    
    // Saving the binarized image.
    imwrite(OUT_PATH+"Binary_image.jpg", this->binary_img);
}

// Finds contours of connected components.
void LineSegmentation::find_contours() {
    cv::Mat img_clone = this->binary_img; // Working on a copy of the binary image.

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    // Finding all contours in the iamge (RETR_LIST gets all contours, no hierarchy).
    cv::findContours(img_clone, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE, cv::Point(0, 0));

    // Approximating contours with polygons and getting bouding rectagles.
    vector<vector<Point> > contours_poly(contours.size());
    vector<Rect> bound_rect(contours.size() - 1);

    // Getting rectangular boundaries from contours.
    for (size_t i = 0; i < contours.size() - 1; i++) {
        // Polygon approximation.
        approxPolyDP(Mat(contours[i]), contours_poly[i], 1, true); 
        // Getting the bounding box.
        bound_rect[i] = boundingRect(Mat(contours_poly[i]));
    }

    // Merging the overlapping/intersecting rectangles.
    Rect2d rectangle3;
    vector<Rect> merged_rectangles;
    bool is_repeated;
    Mat drawing = this->color_img.clone();

    // Checking for intersecting rectangles, and merging them.
    for (int i = 0; i < bound_rect.size(); i++) {
        is_repeated = false;

        for (int j = i + 1; j < bound_rect.size(); j++) {
            // Intersection area.
            rectangle3 = bound_rect[i] & bound_rect[j];

            // If one rectangle completely contains another, merge them.
            if ((rectangle3.area() == bound_rect[i].area()) || (rectangle3.area() == bound_rect[j].area())) {
                is_repeated = true;
                rectangle3 = bound_rect[i] | bound_rect[j]; // Union of rectangles.
                Rect2d merged_rectangle(rectangle3.tl().x, rectangle3.tl().y, 
                                        rectangle3.width, rectangle3.height);

                // Adding merged rectangle at the end of processing.
                if (j == bound_rect.size() - 2)
                    merged_rectangles.push_back(merged_rectangle);

                // Updating the current rectangle.
                bound_rect[j] = merged_rectangle;
            }
        }

        // Adding the non repeated (not intersected) rectangles.
        if (!is_repeated)
            merged_rectangles.push_back(bound_rect[i]);
    }
    
    // Visualizing contours for debugging.
    for (size_t i = 0; i < merged_rectangles.size(); i++)
        rectangle(drawing, merged_rectangles[i].tl(), merged_rectangles[i].br(), TEST_LINE_COLOR, 2, 8, 0);
    cv::imwrite(OUT_PATH+"contours.jpg", drawing);

    this->contours = merged_rectangles; // Storing the final merged contours.
}

// Divides image into vertical chunks for parallel processing.
void LineSegmentation::generate_chunks() {
    int width = binary_img.cols;
    chunk_width = width / CHUNKS_NUMBER; // Calculating the width of each chunk.


    for (int i_chunk = 0, start_pixel = 0; i_chunk < CHUNKS_NUMBER; ++i_chunk) {
        Chunk *c = new Chunk(i_chunk, start_pixel, chunk_width, 
                             cv::Mat(binary_img, cv::Range(0, binary_img.rows), 
                             cv::Range(start_pixel, start_pixel + chunk_width)));

        this->chunks.push_back(c); 

        // Debugging images of the chunks!
        // imwrite(OUT_PATH+"Chunk" + to_string(i_chunk) + ".jpg", this->chunks.back()->img);

        start_pixel += chunk_width; // Moving to the next chunk position.
    }
}

// Recursively connects valleys across chunks to form text lines.
Line *LineSegmentation::connect_valleys(int i, Valley *current_valley, Line *line, int valleys_min_abs_dist) {
    if (i <= 0 || chunks[i]->valleys.empty()) return line;

    // Finding the closest valley in the right chunk to connect to.
    int connected_to = -1;
    int min_distance = 100000;
    for (int j = 0; j < this->chunks[i]->valleys.size(); j++) {
        Valley *valley = this->chunks[i]->valleys[j];
        if (valley->used) continue; // Skipping already used valleys.

        // Calculating vertical distance between valleys.
        int dist = current_valley->position - valley->position;
        dist = dist < 0 ? -dist : dist;

        // Finding the minimum distance within allowed threshold.
        if (min_distance > dist && dist <= valleys_min_abs_dist) {
            min_distance = dist, connected_to = j;
        }
    }

    // If no suitable valley is found, return the current line.
    if (connected_to == -1) {
        return line;
    }

    // Connecting to the found valley and continuing connecting leftwards.
    line->valleys_ids.push_back(this->chunks[i]->valleys[connected_to]->valley_id);
    Valley *v = this->chunks[i]->valleys[connected_to];
    v->used = true;

    // Recursive call.
    return connect_valleys(i - 1, v, line, valleys_min_abs_dist);
}

// Detects initial text lines by connecting valleys across chunks.
void LineSegmentation::get_initial_lines() {
    int number_of_heights = 0, valleys_min_abs_dist = 0;

    // Estimating average line height from first few chunks.
    for (int i = 0; i < CHUNKS_TO_BE_PROCESSED; i++) {
        int avg_height = this->chunks[i]->find_peaks_valleys(map_valley);
        if (avg_height) number_of_heights++;
        valleys_min_abs_dist += avg_height;
    }
    valleys_min_abs_dist /= number_of_heights;
    cout << "Estimated avg line height " << valleys_min_abs_dist << endl;
    this->predicted_line_height = valleys_min_abs_dist;
    
    // Added some tolerance.
    // valleys_min_abs_dist = valleys_min_abs_dist * 1.2;

    // Starting from the rightmost processed chunk and connecting valleys leftwards.
    for (int i = CHUNKS_TO_BE_PROCESSED - 1; i >= 0; i--) {
        if (i<0 || i>=chunks.size()) continue; 
        if (chunks[i]->valleys.empty()) continue;

        // Processing each valley in the current chunk.
        for (auto &valley : chunks[i]->valleys) {
            if (valley->used) continue; // Skipping used valleys.

            // Marking the current valley as read.
            valley->used = true;

            // Creating a new line starting from this valley.
            Line *new_line = new Line(valley->valley_id);
            new_line = connect_valleys(i - 1, valley, new_line, valleys_min_abs_dist);
            new_line->generate_initial_points(chunk_width, color_img.cols, map_valley);

            // Only keeping lines with multiple valleys (more reliable).
            if (new_line->valleys_ids.size() > 1)
                this->initial_lines.push_back(new_line);
        }
    }
}

// Visualizes detected lines on the original image.
void LineSegmentation::save_image_with_lines(string path) {
    cv::Mat img_clone = this->color_img.clone();

    // Draws each line point by point.
    for (auto line : initial_lines) {
        int last_row = -1;
        for (auto point : line->points) {
            img_clone.at<Vec3b>(point.x, point.y) = TEST_LINE_COLOR;
                
            // Draws vertical connections between points in different rows.
            if (last_row != -1 && point.x != last_row) {
                for (int i = min(last_row, point.x); i < max(last_row, point.x); i++) {
                    img_clone.at<Vec3b>(i, point.y) = TEST_LINE_COLOR;
                }
            }

            last_row = point.x;
        }
    }
    // Saving the visualization.
    cv::imwrite(path, img_clone);
}

// Saves individual line images to files.
void LineSegmentation::save_lines_to_file(const vector<cv::Mat> &lines) {
    int idx = 0;
    for (auto m : lines) {
        imwrite(OUT_PATH + "Line_" + to_string(idx++) + ".jpg", m);
    }
}

// Creates and saves a labeld image where each line region has a unique 8--bit value.
void LineSegmentation::save_label_image(const std::vector<cv::Mat> &regions, const std::string &out_path) {
    fs::path target;
    target = fs::path(OUT_PATH + "8bit");
    if (!target.has_extension()) target += ".png";

    // Determining the image size from available sources.
    cv::Size sz;
    if (!this->binary_img.empty()) sz = this->binary_img.size();
    else if (!this->color_img.empty()) sz = this->color_img.size();
    else if (!regions.empty()) sz = regions[0].size();
    else {
        std::cerr << "[save_label_image] No input image or regions available to determine size.\n";
        return;
    }

    // 0 = background, 1 = line0, 2 = line1, ...
    cv::Mat labels = cv::Mat::zeros(sz, CV_8U); 

    // Labels based on the final lines.
    if (!this->outFinal_lines.empty()) {
        std::cout << "[save_label_image] encoding using finalLines (" << this->outFinal_lines.size() << " lines)\n";
        for (size_t li = 0; li < this->outFinal_lines.size(); ++li) {
            Line *line = this->outFinal_lines[li];
            if (!line) continue;
            if (line->points.empty()) {continue;}

            int label_value = static_cast<int>(li);
            uint8_t lab8 = label_value <= 255 ? static_cast<uint8_t>(label_value) : 255;

            // Drawing the line with its unique label value.
            int last_row = -1;
            for (size_t p = 0; p < line->points.size(); ++p) {
                Point pt = line->points[p];
                int r = pt.x;
                int c = pt.y;

                // Bounds check.
                if (r < 0 || r >= sz.height || c < 0 || c >= sz.width) continue;

                // Sets pixel value and connects vertically.
                labels.at<uchar>(r, c) = lab8;
                if (last_row != -1 && last_row != r) {
                    int r0 = std::min(last_row, r);
                    int r1 = std::max(last_row, r);
                    for (int rr = r0; rr <= r1; ++rr) {
                        if (rr < 0 || rr >= sz.height) continue;
                        labels.at<uchar>(rr, c) = lab8;
                    }
                }
                last_row = r;
            }

            std::cout << "[save_label_image] outFinal_lines[" << li << "] labeled pixels: " 
                      << cv::countNonZero(labels == lab8) << "\n";
        }
    // Fallback: use region masks if final lines aren't available.
    } else if (!regions.empty()) {
        std::cout << "[save_label_image] outFinal_lines empty, falling back to regions vector (" << regions.size() << ")\n";
        for (size_t i = 0; i < regions.size(); ++i) {
            cv::Mat mask = regions[i];
            if (mask.size() != sz) cv::resize(mask, mask, sz, 0, 0, cv::INTER_NEAREST);

            cv::Mat gray;
            if (mask.channels() > 1) cv::cvtColor(mask, gray, cv::COLOR_BGR2GRAY);
            else gray = mask;

            cv::Mat bin;
            cv::threshold(gray, bin, 0, 255, cv::THRESH_BINARY);

            int label_value = int(i) + 1;
            uint8_t lab8 = label_value <= 255 ? static_cast<uint8_t>(label_value) : 255;
            if (label_value > 255)
                std::cerr << "[save_label_image] Warning: region index " << i << " exceeds 254 and will be clamped to 255\n";

            // Only assign to unlabeled pixels so regions don't overwrite earlier ones.
            cv::Mat unlabeled = (labels == 0);
            cv::Mat assignMask;
            cv::bitwise_and(bin, unlabeled, assignMask);

            labels.setTo(lab8, assignMask);
            std::cout << "[save_label_image] region " << i+1 << " assigned pixels: " << cv::countNonZero(assignMask) << "\n";
        }
    } else {
        std::cerr << "[save_label_image] No lines or regions to encode.\n";
    }

    double minV, maxV; cv::minMaxLoc(labels, &minV, &maxV);
    std::cout << "[save_label_image] final label range: " << minV << " .. " << maxV << "\n";

    // Writing the 8-bit PNG.
    try {
        if (cv::imwrite(target.string(), labels)) {
            std::cout << "[save_label_image] labels saved (8-bit).\n";
        } else {
            std::cerr << "[save_label_image] imwrite returned false.\n";
        }
    } catch (const cv::Exception &e) {
        std::cerr << "[save_label_image] cv::Exception: " << e.what() << "\n";
    } catch (...) {
        std::cerr << "[save_label_image] unknown exception while writing labels\n";
    }
}

// Created a coloured label image showing different line regions.
void LineSegmentation::labelImage(string path) {
    if (initial_lines.empty()) {
        std::cerr << "No initial lines found. Skipping labelImage." << std::endl;
        return;
    }

    cv::Mat img_clone = this->color_img.clone();
    vector<cv::Point> pointactualLine(initial_lines[0]->points.size());
    vector<cv::Point> pointnextLine;
    
    try {
        // Processes each line and the region below it.
        for (int indexLine = 0; indexLine < initial_lines.size(); indexLine++) {
            if (!initial_lines[indexLine]) {
                std::cerr << "Warning: initial_lines[" << indexLine << "] is nullptr. Skipping." << std::endl;
                continue;
            }
            
            if (initial_lines[indexLine]->points.empty()) {
                std::cerr << "Warning: Line " << indexLine << " has no points. Skipping." << std::endl;
                continue;
            }
            
            std::cout << "indexLine=" << indexLine << ", points=" << initial_lines[indexLine]->points.size() << std::endl;
            pointnextLine = initial_lines[indexLine]->points;
            
            // Ensuring both lines have the same number of points for proper processing.
            if (pointactualLine.size() != pointnextLine.size()) {
                std::cerr << "Warning: Line " << indexLine << " has different point count (" 
                          << pointnextLine.size() << " vs " << pointactualLine.size() 
                          << "). Resizing may cause issues." << std::endl;
                
                // Using the smaller size to avoid out-of-bounds access.
                int min_size = std::min(pointactualLine.size(), pointnextLine.size());
                vector<cv::Point> actualTrimmed(pointactualLine.begin(), pointactualLine.begin() + min_size);
                vector<cv::Point> nextTrimmed(pointnextLine.begin(), pointnextLine.begin() + min_size);
                
                this->labelComponent(nextTrimmed, actualTrimmed, img_clone);
            } else {
                this->labelComponent(pointnextLine, pointactualLine, img_clone);
            }
            
            pointactualLine = pointnextLine;
        }

        // Processing the final "line" (right edge of image) - with bounds checking.
        if (!pointnextLine.empty()) {
            vector<cv::Point> finalLine(pointnextLine.size());
            for(int i = 0; i < pointnextLine.size(); i++) {
                int x = std::min(this->color_img.cols, pointnextLine[i].x + 100); // Add some padding
                finalLine[i] = cv::Point(x, pointnextLine[i].y);
            }
            this->labelComponent(finalLine, pointactualLine, img_clone);
        }

        // Using PNG instead of BMP for better quality and compression.
        string ext = path.substr(path.find_last_of(".") + 1);
        if (ext == "bmp") {
            path = path.substr(0, path.find_last_of(".")) + ".png";
        }
        
        cv::imwrite(path, img_clone);
        std::cout << "Label image saved successfully: " << path << std::endl;
        
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV exception in labelImage: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception in labelImage: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown exception in labelImage" << std::endl;
    }
}

// Colors the region between two lines with a random colour.
void LineSegmentation::labelComponent(const std::vector<cv::Point> &pointnextLine, const vector<cv::Point> &pointactualLine, cv::Mat &img_clone){
    int min_size = std::min(pointactualLine.size(), pointnextLine.size());
    int max = 220;
    int min = 30;
    cv::Vec3b randomColor = cv::Vec3b(rand() % max + min,rand() % max + min, redStart);
    
    for (int indexPoint = 0; indexPoint < min_size; indexPoint++) {
        auto nextPoint = pointnextLine[indexPoint];
        auto point     = pointactualLine[indexPoint];

        for (int x_ = point.x; x_ < nextPoint.x; x_++) {
            if (x_ >= 0 && x_ < img_clone.rows && indexPoint >= 0 && indexPoint < img_clone.cols) {
                if (img_clone.at<Vec3b>(x_, indexPoint)[2] < 125)
                    img_clone.at<Vec3b>(x_, indexPoint) = randomColor;
            }
        }
        redStart += 5;
    }
}

// Creates regions between detected text lines.
void LineSegmentation::generate_regions() {
    // Sorts lines by row position.
    sort(this->initial_lines.begin(), this->initial_lines.end(), Line::comp_min_row_position);

    this->line_regions = vector<Region *>();
    
    // じゅうん: Fixed initialization bug.
    this->avg_line_height = 0;

    // Creates first region (above first line).
    Region *r = new Region(nullptr, this->initial_lines[0]);
    r->update_region(this->binary_img, 0);
    this->initial_lines[0]->above = r;
    this->line_regions.push_back(r);
    if (r->height < this->predicted_line_height * 2.5)
        this->avg_line_height += r->height;

    // Creates regions between lines.
    for (int i = 0; i < this->initial_lines.size(); ++i) {
        Line *top_line = this->initial_lines[i];
        Line *bottom_line = (i == this->initial_lines.size() - 1) ? nullptr : this->initial_lines[i + 1];

        // Assigns lines to region.
        Region *r = new Region(top_line, bottom_line);
        bool res = r->update_region(this->binary_img, i);

        if (top_line != nullptr)
            top_line->below = r;

        if (bottom_line != nullptr)
            bottom_line->above = r;

        if (!res) {
            this->line_regions.push_back(r);
            if (r->height < this->predicted_line_height * 2.5)
                this->avg_line_height += r->height;
        }
    }

    // Calculates the average line height.
    if (this->line_regions.size() > 0) {
        this->avg_line_height /= this->line_regions.size();
        cout << "Avg line height is " << this->avg_line_height << endl;
    }
}

// Repairs lines by adjusting them based on character contours. 
void LineSegmentation::repair_lines() {
    // Processes each line.
    for (Line *line : initial_lines) {
        map<int, bool> column_processed = map<int, bool>();

        // Processes each point in the line.
        for (int i = 0; i < line->points.size(); i++) {
            Point &point = line->points[i];

            int x = (line->points[i]).x, y = (line->points[i]).y;

            // Check for vertical line intersection 
            // (current point on a black pixel).
            if (this->binary_img.at<uchar>(point.x, point.y) == 255) {
                if (i == 0) continue;

                bool black_found = false;
                if (line->points[i - 1].x != line->points[i].x) {
                    // Vertical line segment - checks for text intersection.
                    int min_row = min(line->points[i - 1].x, line->points[i].x);
                    int max_row = max(line->points[i - 1].x, line->points[i].x);

                    for (int j = min_row; j <= max_row && !black_found; ++j) {
                        if (this->binary_img.at<uchar>(j, line->points[i - 1].y) == 0) {
                            x = j, y = line->points[i - 1].y;
                            black_found = true;
                        }
                    }
                }
                if (!black_found) continue;
            }

            // Skips processed columns.
            if (column_processed[y]) continue;
            column_processed[y] = true; // Marks column as processed

            // Checks if the point intersects with any character contour. 
            for (auto contour : this->contours) {
                // Check line & contour intersection
                if (y >= contour.tl().x && y <= contour.br().x && x >= contour.tl().y && x <= contour.br().y) {

                    // If contour is longer than the average height ignore.
                    if (contour.br().y - contour.tl().y > this->avg_line_height * 0.9) continue;

                    // Determine if the contour belongs to the line above or below.
                    bool is_component_above = component_belongs_to_above_region(*line, contour);

                    int new_row;
                    if (!is_component_above) {
                        new_row = contour.tl().y; // Move line to the top of the contour.
                        line->min_row_position = min(line->min_row_position, new_row);
                    } else {
                        new_row = contour.br().y; // Move line to the bottom of the contour.
                        line->max_row_position = max(new_row, line->max_row_position);
                    }

                    // Adjuts line points for this column range.
                    for (int k = contour.tl().x; k < contour.tl().x + contour.width; k++) {
                        line->points[k].x = new_row;
                    }
                    i = (contour.br().x); // Skips processed columns.

                    break; // Contour found
                }
            }
        }
    }
}

// Determines if a contour belongs to the region above or below a line using probabilities.
bool LineSegmentation::component_belongs_to_above_region(Line &line, Rect &contour) {
    // Uses prime factorization for probability comparison.
    vector<int> probAbovePrimes(primes.size(), 0);
    vector<int> probBelowPrimes(primes.size(), 0);
    int n = 0;

    // Calculates probabilities for each pixel in the contour.
    for (int i_contour = contour.tl().x; i_contour < contour.tl().x + contour.width; i_contour++) {
        for (int j_contour = contour.tl().y; j_contour < contour.tl().y + contour.height; j_contour++) {
            // Skips white pixels.
            if (binary_img.at<uchar>(j_contour, i_contour) == 255) continue;

            n++;
            Mat contour_point = Mat::zeros(1, 2, CV_32F);
            contour_point.at<float>(0, 0) = j_contour;
            contour_point.at<float>(0, 1) = i_contour;

            // Gets probabilities using Gaussian density models.
            int newProbAbove = (int) ((line.above != nullptr) ? (line.above->bi_variate_gaussian_density(
                    contour_point.clone())) : 0);
            int newProbBelow = (int) ((line.below != nullptr) ? (line.below->bi_variate_gaussian_density(
                    contour_point.clone())) : 0);

            // Factorizes probabilities into prime components.
            addPrimesToVector(newProbAbove, probAbovePrimes);
            addPrimesToVector(newProbBelow, probBelowPrimes);
        }
    }

    // Reconstructs probabilities from prime factors.
    int prob_above = 0, prob_below = 0;
    for (int k = 0; k < probAbovePrimes.size(); ++k) {
        int mini = min(probAbovePrimes[k], probBelowPrimes[k]);

        probAbovePrimes[k] -= mini;
        probBelowPrimes[k] -= mini;

        prob_above += probAbovePrimes[k] * primes[k];
        prob_below += probBelowPrimes[k] * primes[k];
    }

    /// Contour belongs to region with the higher probability.
    return prob_above < prob_below;
}

vector<cv::Mat> LineSegmentation::segment() {
    // Cleaning output directory before running.
    std::cout << "Output path: " << OUT_PATH << std::endl;
    if (!OUT_PATH.empty()) {
        try {
            fs::path outp(OUT_PATH);
            fs::path img_dir = outp.parent_path(); // if OUT_PATH == "img/out" -> img_dir == "img"
            if (img_dir.empty()) img_dir = ".";

            // Removing existing image files.
            if (fs::exists(img_dir) && fs::is_directory(img_dir)) {
                const std::vector<std::string> exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"};
                for (auto &entry : fs::directory_iterator(img_dir)) {
                    try {
                        if (!fs::is_regular_file(entry.path())) continue;
                        std::string ext = entry.path().extension().string();
                        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                        if (std::find(exts.begin(), exts.end(), ext) != exts.end()) {
                            fs::remove(entry.path());
                            std::cout << "[cleanup] removed image: " << entry.path().string() << "\n";
                        }
                    } catch (const std::exception &e) {
                        std::cerr << "[cleanup] failed to remove " << entry.path().string()
                                  << ": " << e.what() << "\n";
                    }
                }
            } else {
                // Creating img_dir (parent) if it doesn't exist.
                fs::create_directories(img_dir);
                std::cout << "[cleanup] created directory: " << img_dir.string() << "\n";
            }
        } catch (const std::exception &e) {
            std::cerr << "[cleanup] exception while clearing img folder: " << e.what() << "\n";
        }
    }
    
    // Reads image and process it.
    std::cout << "Starting image processing." << std::endl;
    this->read_image();
    this->pre_process_image();

    // Finds letters contours.
    std::cout << "Finding letter contours." << std::endl;
    this->find_contours();

    // Divides image into vertical chunks.
    std::cout << "Dividing into vertical chunks." << std::endl;
    this->generate_chunks();

    // Gets initial lines.
    std::cout << "Getting initial lines." << std::endl;
    this->get_initial_lines();
    this->save_image_with_lines(OUT_PATH+"Initial_Lines.jpg");

    // Gets initial line regions.
    std::cout << "Getting initial line regions.." << std::endl;
    this->generate_regions();

    // Repairs initial lines and generate the final line regions.
    std::cout << "Repairing lines.." << std::endl;
    this->repair_lines();

    // Generates the final line regions.
    std::cout << "Generating the final regions." << std::endl;
    this->generate_regions();

    // Saves the final image.
    std::cout << "Saving the image with lines.." << std::endl;
    this->save_image_with_lines(OUT_PATH+"Final_Lines.bmp");

    // Is neccesary to use bitmap or tiff for component labeling.
    std::cout << "Adding labeling." << std::endl;
    this->labelImage(OUT_PATH+"labels.bmp");

    // Saving an 8-bit encoding.
    //std::cout << "Saving 8-bit encoding." << std::endl;
    auto regions = this->get_regions();
    //this->save_label_image(regions, "img/out/labels.png");

    return regions;
}

// Returns the segmented line regions.
vector<cv::Mat> LineSegmentation::get_regions() {
    vector<cv::Mat> ret;
    for (auto region : this->line_regions) {
        ret.push_back(region->region.clone());
    }
    return ret;
}

// Chunk constructor - represents a vertical slice of the image.
Chunk::Chunk(int i, int c, int w, cv::Mat m)
        : valleys(vector<Valley *>()), peaks(vector<Peak>()) {
    this->index = i;         // Chunk index.
    this->start_col = c;     // Starting column.
    this->width = w;         // Width of chunk.
    this->img = m.clone();   // Image data for this chunk.
    this->histogram.resize((unsigned long) this->img.rows); // Vertical projection histogram.
    this->avg_height = 0;    // Average line height in chunk.
    this->avg_white_height = 0; // Average white space height.
    this->lines_count = 0;   // Number of lines detected.
}

// Calculates the vertical projection histogram for the chunk.
void Chunk::calculate_histogram() {
    // Smoothes the image with median filter to reduce noise.
    cv::Mat img_clone;
    cv::medianBlur(this->img, img_clone, 5);
    this->img = img_clone;

    // Calculates histogram and detects lines/white spaces.
    int black_count = 0, current_height = 0, current_white_count = 0, white_lines_count = 0;
    vector<int> white_spaces;

    for (int i = 0; i < img_clone.rows; ++i) {
        black_count = 0;
        for (int j = 0; j < img_clone.cols; ++j) {
            if (img_clone.at<uchar>(i, j) == 0) {
                black_count++;
                this->histogram[i]++;
            }
        }

        // Tracks line heights and white spaces. 
        if (black_count) {
            current_height++;
            if (current_white_count) {
                white_spaces.push_back(current_white_count);
            }
            current_white_count = 0;
        } else {
            current_white_count++;
            if (current_height) {
                lines_count++;
                avg_height += current_height;
            }
            current_height = 0;
        }
    }

    // Calculates average white space height (excluding outliers).
    sort(white_spaces.begin(), white_spaces.end());
    for (int i = 0; i < white_spaces.size(); ++i) {
        if (white_spaces[i] > 4 * avg_height) break;
        avg_white_height += white_spaces[i];
        white_lines_count++;
    }
    if (white_lines_count) avg_white_height /= white_lines_count;

    // Calculate the average line height.
    if (lines_count) avg_height /= lines_count;

    // Applying minimum threshold.
    avg_height = max(30, int(avg_height + (avg_height / 2.0))); 
}

// Finds peaks (text lines) and valleys (between lines) in the chunk.
int Chunk::find_peaks_valleys(map<int, Valley *> &map_valley) {
    this->calculate_histogram();

    // Detects peaks in the histogram (local maxima).
    for (int i = 1; i + 1 < this->histogram.size(); i++) {
        int left_val = this->histogram[i - 1], centre_val = this->histogram[i], right_val = this->histogram[i + 1];
        if (centre_val >= left_val && centre_val >= right_val) { // Peak detected.
            // Merges nearby peaks, keeping the strongest one.
            if (!peaks.empty() && i - peaks.back().position <= avg_height / 2 &&
                centre_val >= peaks.back().value) { 
                peaks.back().position = i;
                peaks.back().value = centre_val;
            } else if (peaks.size() > 0 && i - peaks.back().position <= avg_height / 2 &&
                       centre_val < peaks.back().value) {}
            else {
                peaks.push_back(Peak(i, centre_val));
            }
        }
    }

    // Filters out weak points.
    int peaks_average_values = 0;
    vector<Peak> new_peaks;
    for (auto peak : peaks) {
        peaks_average_values += peak.value;
    }
    peaks_average_values /= max(1, int(peaks.size()));

    // Keeps peaks above threshold.
    for (auto peak : peaks) {
        if (peak.value >= peaks_average_values / 4) {
            new_peaks.push_back(peak);
        }
    }

    lines_count = int(new_peaks.size());
    peaks = new_peaks;

    // Sorts and limits the number of peaks.
    sort(peaks.begin(), peaks.end());
    peaks.resize(lines_count + 1 <= peaks.size() ? (unsigned long) lines_count + 1 : peaks.size());
    sort(peaks.begin(), peaks.end(), Peak::comp); // Sort peaks by position.

    // Finds valleys between peaks (minimum points between lines).
    for (int i = 1; i < peaks.size(); i++) {
        pair<int, int> expected_valley_positions[4];
        int min_position = (peaks[i - 1].position + peaks[i].position) / 2;
        int min_value = this->histogram[min_position];

        // Searches for the actual valley position.
        for (int j = (peaks[i - 1].position + avg_height / 2);
             j < (i == peaks.size() ? this->img.rows : peaks[i].position - avg_height - 30); j++) {

            int valley_black_count = 0;
            for (int l = 0; l < this->img.cols; ++l) {
                if (this->img.at<uchar>(j, l) == 0) {
                    valley_black_count++;
                }
            }

            // Updates the minimum position.
            if (i == peaks.size() && valley_black_count <= min_value) {
                min_value = valley_black_count;
                min_position = j;
                if (!min_value) { // Found perfect value (no text).
                    min_position = min(this->img.rows - 10, min_position + avg_height);
                    j = this->img.rows;
                }
            } else if (min_value != 0 && valley_black_count <= min_value) {
                min_value = valley_black_count;
                min_position = j;
            }
        }

        // Creating and storing the valley.
        auto *new_valley = new Valley(this->index, min_position);
        valleys.push_back(new_valley);
        map_valley[new_valley->valley_id] = new_valley;
    }

    return int(ceil(avg_height)); // Returns the estimated line height.
}

// Line constructor - represents a text line.
Line::Line(int initial_valley_id)
        : min_row_position(0), max_row_position(0), points(vector<Point>()) {
    valleys_ids.push_back(initial_valley_id);
}

// Generates points along the line by connecting valleys across chunks.
void Line::generate_initial_points(int chunk_width, int img_width, map<int, Valley *> map_valley) {
    int c = 0, previous_row = 0;

    // Sort the valleys according to their chunk number.
    sort(valleys_ids.begin(), valleys_ids.end());

    // Add line points in the first chunks having no valleys.
    if (map_valley[valleys_ids.front()]->chunk_index > 0) {
        previous_row = map_valley[valleys_ids.front()]->position;
        max_row_position = min_row_position = previous_row;
        for (int j = 0; j < map_valley[valleys_ids.front()]->chunk_index * chunk_width; j++) {
            if (c++ == j)
                points.push_back(Point(previous_row, j));
        }
    }

    // Add line points between the valleys.
    for (auto id : valleys_ids) {
        int chunk_index = map_valley[id]->chunk_index;
        int chunk_row = map_valley[id]->position;
        int chunk_start_column = chunk_index * chunk_width;

        for (int j = chunk_start_column; j < chunk_start_column + chunk_width; j++) {
            min_row_position = min(min_row_position, chunk_row);
            max_row_position = max(max_row_position, chunk_row);
            if (c++ == j)
                points.push_back(Point(chunk_row, j));
        }
        if (previous_row != chunk_row) {
            previous_row = chunk_row;
            min_row_position = min(min_row_position, chunk_row);
            max_row_position = max(max_row_position, chunk_row);
        }
    }

    // Adds points in the last chunks having no valleys.
    if (CHUNKS_NUMBER - 1 > map_valley[valleys_ids.back()]->chunk_index) {
        int chunk_index = map_valley[valleys_ids.back()]->chunk_index,
                chunk_row = map_valley[valleys_ids.back()]->position;
        for (int j = chunk_index * chunk_width + chunk_width; j < img_width; j++) {
            if (c++ == j)
                points.push_back(Point(chunk_row, j));
        }
    }
}

// Comparison for sorting lines by vertical position.
bool Line::comp_min_row_position(const Line *a, const Line *b) {
    return a->min_row_position < b->min_row_position;
}

// Peak comparison in operators.
bool Peak::operator<(const Peak &p) const {
    return value > p.value;
}

bool Peak::comp(const Peak &a, const Peak &b) {
    return a.position < b.position;
}

// Valley static ID counter and comparison.
int Valley::ID = 0;

bool Valley::comp(const Valley *a, const Valley *b) {
    return a->position < b->position;
}

// Region constructor - represents space between lines.
Region::Region(Line *top, Line *bottom) {
    this->top = top;
    this->bottom = bottom;
    this->height = 0;
}

// Extracts the image region between top and bottom lines.
bool Region::update_region(Mat &binary_image, int region_id) {
    this->region_id = region_id;

    int min_region_row = row_offset = (top == nullptr) ? 0 : top->min_row_position;
    int max_region_row = (bottom == nullptr) ? binary_image.rows : bottom->max_row_position;

    int start = min(min_region_row, max_region_row), end = max(min_region_row, max_region_row);

    // Creates white region mask.
    region = Mat::ones(end - start, binary_image.cols, CV_8U) * 255;

    // Extracts the actual text content for this region.
    for (int c = 0; c < binary_image.cols; c++) {
        int start = ((top == nullptr) ? 0 : top->points[c].x);
        int end = ((bottom == nullptr) ? binary_image.rows - 1 : bottom->points[c].x);

        // Calculates region height.
        if (end > start)
            this->height = max(this->height, end - start);

        // Copies pixels from binary image to region.
        for (int i = start; i < end; i++) {
            region.at<uchar>(i - min_region_row, c) = binary_image.at<uchar>(i, c);
        }
    }

    // Calculate statistics for the region.
    calculate_mean();
    calculate_covariance();

    // Returns true if region is empty (all white).
    return countNonZero(region) == region.cols * region.rows;
}

// Calculates mean position of black pixels in the region.
void Region::calculate_mean() {
    mean[0] = mean[1] = 0.0f;
    int n = 0;
    for (int i = 0; i < region.rows; i++) {
        for (int j = 0; j < region.cols; j++) {
            // if white pixel continue.
            if (region.at<uchar>(i, j) == 255) continue;
            if (n == 0) {
                n = n + 1;
                mean = Vec2f(i + row_offset, j);
            } else {
                mean = (n - 1.0) / n * mean + 1.0 / n * Vec2f(i + row_offset, j);
                n = n + 1;
            }
        }
    }
}

// Calculates covariance matrix of black pixel positions.
void Region::calculate_covariance() {
    Mat covariance = Mat::zeros(2, 2, CV_32F);

    int n = 0; // Total number of considered points so far.
    float sum_i_squared = 0, sum_j_squared = 0, sum_i_j = 0;

    for (int i = 0; i < region.rows; i++) {
        for (int j = 0; j < region.cols; j++) {
            // if white pixel continue.
            if ((int) region.at<uchar>(i, j) == 255) continue;

            float new_i = i + row_offset - mean[0];
            float new_j = j - mean[1];

            sum_i_squared += new_i * new_i;
            sum_i_j += new_i * new_j;
            sum_j_squared += new_j * new_j;
            n++;
        }
    }

    // Normalizes by number of points.
    if (n) {
        covariance.at<float>(0, 0) = sum_i_squared / n;
        covariance.at<float>(0, 1) = sum_i_j / n;
        covariance.at<float>(1, 0) = sum_i_j / n;
        covariance.at<float>(1, 1) = sum_j_squared / n;

    }

    this->covariance = covariance.clone();
}

// Calculates Gaussian probability density for a point in this region.
double Region::bi_variate_gaussian_density(Mat point) {
    // Center point around mean.
    point.at<float>(0, 0) -= this->mean[0];
    point.at<float>(0, 1) -= this->mean[1];

     // Calculates Mahalanobis distance.
    Mat point_transpose;
    transpose(point, point_transpose);
    Mat ret = ((point * this->covariance.inv() * point_transpose));
    ret *= sqrt(determinant(this->covariance * 2 * M_PI));

    return ret.at<float>(0, 0);
}

// Sieve of Eratosthenes for prime number generation.
void LineSegmentation::sieve() {
    not_primes_arr[0] = not_primes_arr[1] = 1;
    for (int i = 2; i < 1e5; ++i) {
        if (not_primes_arr[i]) continue;

        primes.push_back(i);
        for (int j = i * 2; j < 1e5; j += i) {
            not_primes_arr[j] = 1;
        }
    }
}

// Factorizes a number into prime components for probability comparison.
void LineSegmentation::addPrimesToVector(int n, vector<int> &probPrimes) {
    for (int i = 0; i < primes.size(); ++i) {
        while (n % primes[i]) {
            n /= primes[i];
            probPrimes[i]++;
        }
    }
}
