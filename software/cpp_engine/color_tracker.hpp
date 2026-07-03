#ifndef COLOR_TRACKER_HPP
#define COLOR_TRACKER_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <map>

struct BoundingBox {
    int x;
    int y;
    int w;
    int h;
};

struct BlobResult {
    BoundingBox bbox;
    int cx;
    int cy;
    double area;
};

class ColorTracker {
public:
    ColorTracker();
    
    // Process a planar YUV frame natively
    // Returns a dictionary-like structure of detected elements
    std::map<std::string, std::vector<BlobResult>> process_yuv_frame(
        const cv::Mat& y_channel, 
        const cv::Mat& u_channel, 
        const cv::Mat& v_channel,
        int scale,
        int cam_id
    );

private:
    // Core color masking bounds (y_min, y_max, u_min, u_max, v_min, v_max)
    int green_bounds[6];
    int orange_bounds[6];
    int blue_bounds[6];
    int yellow_bounds[6];
    int black_bounds[6];
    int white_bounds[6];

    double min_ball_area_px;
    double min_enemy_area_px;

    // Masking geometry constraints
    double fisheye_rx_pct;
    double fisheye_ry_pct;
    double crop_top_pct;
    double crop_bottom_pct;
    
    cv::Mat get_static_lens_mask(int h, int w);
    
    // Caching for performance
    std::map<std::pair<int, int>, cv::Mat> mask_cache;
};

#endif // COLOR_TRACKER_HPP
