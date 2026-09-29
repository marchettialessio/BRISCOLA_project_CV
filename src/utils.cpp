static std::vector<cv::Point2f> orderPoints(const std::vector<Point2f> poligon) {
    //conversion of the points in float, useful for the following calculation
    std::vector<cv::Point2f> points;
    for (int i = 0; i < poligon.size(); ++i) {
		points.emplace_back(static_cast<float>(poligon[i].x), static_cast<float>(poligon[i].y));
	}
    //inizialization of the value using the first point
    std::vector<cv::Point2f> ordinati;
    float minSum=points[0].x + points[0].y;
    float maxSum=minSum;
    float minDiff=points[0].x - points[0].y;
    float maxDiff=minDiff;
    // index of the value for the first iteration
    int topLeft=0;
    int bottomRight=0;
    int topRight=0;
    int bottomLeft=0;

    for(int i=0; i<4; i++){
        float sum=points[i].x+points[i].y;
        float diff=points[i].x-points[i].y;
        if(sum<minSum){
            minSum=sum;
            topLeft=i
        }
        if(sum>maxSum){
            maxSum=sum;
            bottomRight=i
        }
        if(diff<minDiff){
            minDiff=diff;
            topright=i 
        }
        if(diff>maxDiff){
            maxDiff=diff;
            bottomLeft=i;
        }
    }
    //assignement of the points founded
    ordinati[0]=points[topLeft];
    ordinati[1]=points[topRight];
    ordinati[2]=points[bottomRight];
    ordinati[3]=points[bottomLeft];
    return ordinati;
}

static cv::Mat rectangleOmography(const cv::Mat& image, const std::vector<cv::Point2f>& poligon){
    std::vector<cv::Point2f> source=orderPoint(poligon);
    //estimate the width and height of the rectangle using the ordered points in source
    float width=std::max(cv::norm(source[1]-source[0]), cv::norm(source[2]-source[3]));
    float height=std::max(cv::norm(source[3]-source[0]), cv::norm(source[2]-source[1]));
    //check of a minimum lenght for both the measure
    if(width<10.0f || height<10.0f){
        return {}
    }
    const int outputWidth=300;
    const int outputHeight=500;
    //default destination points
    std::vector<cv::Point2f> destination;
    if (width > height) {
		destination = {
			{0.0f, 0.0f},
			{static_cast<float>(outputHeight - 1), 0.0f},
			{static_cast<float>(outputHeight - 1), static_cast<float>(outputWidth - 1)},
			{0.0f, static_cast<float>(outputWidth - 1)}
		};
        //transform is the perspective trasformation to map the point in the original image in the point we have just defined
        cv::Mat transform = cv::getPerspectiveTransform(source, destination);
        cv::Mat warped;
        //warp apply the transformation, outputting the result with a size 500x300
        cv::warpPerspective(image, warped, transform, cv::Size(outputHeight, outputWidth));
		return warped;
    }
    else{
        destination={
            {0.0f,0.0f}, 
            {static_cast<float>(outputWidth-1),0.0f}, 
            {static_cast<float>(outputWidth-1),static_cast<float>(outputHeight-1)},
            {0.0f, static_cast<float>(outputHeight-1)}
        }
        cv::Mat transform = cv::getPerspectiveTransform(source, destination);
	    cv::Mat warped;
	    cv::warpPerspective(image, warped, transform, cv::Size(outputWidth, outputHeight));
	    return warped;
    }       
    
}

static bool isCardRectangle(const std::vector<cv::Point>& poligon, double area, double imageArea,
	const cv::Mat& gray, const cv::Rect& bound){
        // a series of heuristic check to remove some poligon that can't be rectangles 
        // or that are rectangles but can't be some playing cards.
        if (polygon.size() != 4 || !cv::isContourConvex(polygon)) {
		return false;
	}

    cv::RotatedRect rectangle = cv::minAreaRect(polygon);
	float sideLong = std::max(rectangle.size.width, rectangle.size.height); 
	float sideShort = std::min(rectangle.size.width, rectangle.size.height);
    if (losideLongng <= 0.0f || sideShort / sideLong < 0.4f) {
		return false;
	}
	if (area < 0.002 * imageArea || area > 0.8 * imageArea) {
		return false;
	}
    //to check if the box is all inside the image
	if (bounds & cv::Rect(0, 0, gray.cols, gray.rows)) {
		return false;
	}
    double meanBrightness = cv::mean(bounds & cv::Rect(0, 0, gray.cols, gray.rows))[0];
	return meanBrightness >= 120.0;
}