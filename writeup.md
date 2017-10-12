# **Finding Lane Lines on the Road** 

Jonathan Soo - jcsoo@agora.com

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline is similar to what was described in the project introductory videos:

   - Convert to grayscale

[grayscale]: ./test_images_grayscale/solidWhiteRight.jpg "Grayscale"

   - Run the Canny filter
   - Optionally, Gaussian blur the Canny output
   - Mask the image to a specific region of interest (ROI)

[canny_masked]: ./test_images_canny_masked/solidWhiteRight.jpg "Canny Masked"

   - Use the Hough Transform to identify line candidates

[hough]: ./test_images_canny_masked/solidWhiteRight.jpg "Hough Transform"

   - Filter + group the line candidates
   - Draw the remaining candidates, bounded by the ROI  
   - Merge the lines with the original image

[output]: ./test_images_output/solidWhiteRight.jpg "Output"
   

Most of my work went into filtering and grouping the line candidates, in the group_lines method,
below:

```

def group_lines(lines, vertices):
    # Basic Approach
    #   Convert lines into (slope, intercept, length) form
    #   NOTE: actually using the reciprocal of the slope because most of our lines will be  close to vertical
    #   Bin into (slope, intercept) groups, 
    #   for each bin
    #       filter for items within envelope
    #       length-weighted average of slope, intercept
    #       generate line bound by ROI

    # Calculate bounds from ROI vertices

    y_max = vertices[0][0][1]
    y_min = vertices[0][1][1]

    x_min = vertices[0][0][0]
    x_max = vertices[0][3][0]
    x_mid = int((x_max + x_min) / 2)

    # Add all lines to dictionary using rounded slope and offset as the key. GROUP_SLOPE and GROUP_OFFSET
    # control the amount of rounding.

    clusters = {}

    for line in lines:
        (l, s, o) = len_slope_offset(line)
        key = (round(s * GROUP_SLOPE) / GROUP_SLOPE, round(o / GROUP_OFFSET) * GROUP_OFFSET)
        v = clusters.get(key, [])
        v.append((line, l, s, o))
        clusters[key] = v

    keys = list(clusters.keys())
    keys.sort()

    out = []

    # For each group of lines in the cluster, reject lines outside of envelope and then
    # perform a length-weighted average of slope + offset.

    for k in keys:
        v = clusters[k]
        v_lines = [l[0] for l in v]
        num = len(v)
        sum_len = sum([l[1] for l in v])
        if num >= GROUP_MIN_NUM and sum_len >= GROUP_MIN_SUM:
            s_sum = 0.0
            o_sum = 0.0
            l_sum = 0.0
            for item in v:
                (line, l, s, o) = item
                if GROUP_ITEM_ENABLE_ENVELOPE:
                    if o > x_mid:
                        if o < x_max * GROUP_LEFT_OFFSET_MIN or o > x_max * GROUP_LEFT_OFFSET_MAX:
                            #print('    left reject offset', s_avg, i_avg, sum_len)
                            continue
                        if s < GROUP_LEFT_SLOPE_MIN or s > GROUP_LEFT_SLOPE_MAX:
                            if DEBUG:
                                print('    left reject slope', s, o, l)
                            continue
                    else:
                        # print('  left')
                        if o < x_max * GROUP_RIGHT_OFFSET_MIN or o > x_max * GROUP_RIGHT_OFFSET_MAX:
                            #print('    right reject offset', s_avg, i_avg, sum_len)
                            continue
                        if s < GROUP_RIGHT_SLOPE_MIN or s > GROUP_RIGHT_SLOPE_MAX:
                            if DEBUG:
                                print('    right reject slope', s, o, l)
                            continue


                s_sum += (l * s)
                o_sum += (l * o)
                l_sum += l

            # No items left in this group due to filtering
            if l_sum == 0:
                continue

            s_avg = s_sum / l_sum
            o_avg = o_sum / l_sum

            # Calculate the line from y_min to y_max

            y0 = int(y_min)
            x0 = int((y0 * s_avg) + o_avg)
            y1 = int(y_max)
            x1 = int((y1 * s_avg) + o_avg)

            line = [[x0, y0, x1, y1]]

            # Add it to the output
            out.append(line)   
    return out
```

The first step is to cluster the candidate lines together. In this case, it is
fairly straightforward to do this in cartesian gradient space, with each line
represented by y = mx + b. The inverse of the slope is used because the lines
we are interested are close to vertical, and we want to avoid infinite slopes.

The slope and offset of each line is then rounded and used as a key in the cluster dictionary, 
essentially binning by slope and offset. We can use fairly large bins because there will
usually be only a small number of lane lines that we are interested in, and they are widely 
separated by both slope and offset. In particular, for this assignment it is simpler
to have a single line representing the lane marker rather than a pair of lines
representing the inner and outer edges of the marker.

Once we have binned all of the edges, we then iterate through the groups to consolidate
the edges into a single line. 

First, we apply an envelope filter using the
slope and offset of each edge (this can actually be done before binning, but
there's no difference in the result). We can take advantage of our knowledge
that due to perspective, lines to our right will tend to angle left and
lines on our left will tend to angle right. 

I have also added a similar offset
window for each side, assuming that we are close to centered within a normal
sized lane; this would need to be disabled if the vehicle was changing lanes,
for instance.

If there are no remaining edges, we continue to the next loop iteration here.

We then average the offset and slope for the remaining edges, weighting
by the length of the edge. This assumes that longer edges produced by the Hough
transform are higher quality than shorter ones.

Finally, we use the offset + slope formula to draw a line bounded by the
top and bottom of the ROI.

#### Observations

Before implementing the grouping function, I spent a significant amount of time
tuning the Canny and Hough transform parameters in order to generate a very
"clean" output, with as many long line segments as possible and few noisy
outliers. This worked well some of the images, but not as well for the videos,
particularly the challenge video.

As I implemented the grouping function including the slope + offset window, the
performance improved significantly, but there were still many places (particularly
in the challenge video) where the system would not draw a line for several difficult
frames.

For a self driving car, it's probably more important to have a line detector that is
robust, even if it is noisier, as long as it doesn't produce too many false positives.
I decided to experiment with relaxing the Canny and Hough transform parameters while
adding the slope + offset envelope to filter out the higher number of invalid edges
produced. This also removed the horizontal edges at the bottom of the challenge
image from the hood of the car.

The final version seems to be a good compromise. It's maybe a little bit noisier in
the solidYellow and solidWhite cases, but not significantly. On the challenge
video, it manages to capture lane lines for almost every single frame, even with a
bit of jitter in poor conditions.

### 2. Identify potential shortcomings with your current pipeline

There are a few shortcomings and limitations associated with this pipeline.

First, performance is highly dependent on the slope + offset envelope filter,
which assumes that lane lines will be in a particular configuration. If the camera
is angled beyond what is expected or offset beyond the limits of the lane,
lane lines will be filtered out.

Second, this pipeline (intentionally) is stateless - each frame is processed independently. This
means that lines are not necessarily stable - they can shift by several pixels from
frame to frame if vision conditions are poor.

Third, this pipeline makes no use of color information, so it has no way of distinguishing
between line colors.

Fourth, this pipeline makes no attempt to distinguish between solid and dashed lines.

From a performance perspective, this pipeline is currently tuned to produce a large number of
edges from the Canny filter, which in turns results in a large number of pixels being
fed into the Hough transform. Both of these require a significant amount of CPU and
memory, making the pipeline less efficient than one tuned to produce fewer Canny edges.

### 3. Suggest possible improvements to your pipeline

There are several possible improvements to this pipeline.

The first set of improvements tackle the front-end of the problem - going from the raw image
to the filtered and grouped lane lines.

The Canny edge detector is a general purpose edge detector, and detects edges in all orientations, so
it requires scanning the image vertically as well as horizontally. For this application, we would rather 
have a detector that only finds vertical edges; this could be done in a single pass per line.

The vertical line detector could be tuned to respond only to lines of a certain set of colors and
widths, and could indicate the line color as well as whether it is a leading or trailing edge and the
overall strength of the edge match. The expected line width could be parameterized to vary according
to position in the frame, with wider line widths expected towards the bottom of the frame.

From a programming perspective, the line detector could be designed as an iterator feeding edges directly
into the pipeline rather than building a complete bitmap. This would save a certain amount of memory
and latency.

The current pipeline uses a generic Hough Transform line detector, operating in a radial parameter space. This
is required if the detector needs to detect lines of all orientations, but is unnecessary if
the rough orientation of the lines is limited. In this case, we can use a cartesian parameter space,
perhaps with the x and y axes transposed so that we are searching for near-horizontal lines rather than
near-vertical lines. Binning can take place using Bresenham's line algorithm rather than trigonometric
lookup, and should be much faster.

This also ties in neatly with the filtering and grouping method in the current pipeline, which uses an envelope
in the same parameter space. This could be implemented directly in the transform stage, saving several
steps.

As a side experiment, I implemented a Cartesian Hough line detector in Rust:

```

...

    // Use the canny detector from the imageproc library
    use imageproc::edges::canny;

    let edges = canny(&gray, 50.0, 200.0);

    let center = (dim.0 / 2) as i32;

    let dim = gray.dimensions();

    for y in 0..dim.1 {
        for x in 0..dim.0 {
            let p = edges.get_pixel(x, y)[0];
            if p > 0 {
                // Transpose and recenter

                let x0 = dim.1 as i32 - y as i32;
                let y0 = (dim.0 / 2) as i32 - x as i32;

                let fx0 = x0 as f32;
                let fy0 = y0 as f32;

                let offset = fy0 / fx0;;
                let slope = 1.0 / fx0;

                for b in -512i32 .. 512 {
                    let fb = b as f32;
                    let m = offset + (fb / fx0);
                    // -1.0 <=> 1.0 maps to -512 <=> 512
                    let pm = (m * 128.0) as i32;
                    let pb = b as i32;

                    let tm = (pm + 512) as u32;
                    let tb = (pb + 512) as u32;

                    if  tm < 1024 {
                        let n = out.get_pixel(tb, tm)[0];
                        let n = if n < 65535 {
                            n + 1
                        } else {
                            n
                        };
                        out.put_pixel(tb, tm, Luma([n]));
                    }
                }

            }
        }
    }
    // imageproc function to zero out anything that isn't a local maximum, radius 8 pixels
    let out = suppress_non_maximum(&out, 8);
    ...

```

This particular snippet does a transform of the underlying image so that it is transposed and
shifted to have the original bottom center of the frame as the origin, then converts to 
y = mx + b parameter space, accumulates the line associated with each point, then
retrieves the local maxima. Testing with the sample images from this project produces good
results; I haven't tested it with the videos.