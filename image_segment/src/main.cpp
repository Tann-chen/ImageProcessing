#include <opencv2/opencv.hpp>
#include <vector>
#include <queue>

using namespace cv;
using namespace std;

struct Edge {
    int index_from;
    int index_to;
    int flow_capability;
};

struct Vertex {
    int row;
    int col;
};

int indexInMatrix(int row, int col, int row_len);
Vertex castIndex2Vertex(int index, int row_len);
int weightBetweenPixel(Vec3b p1, Vec3b p2);
vector<Edge> find_augment_path_bfs(vector<vector<Edge> > adj_list, int src, int tar);


int main(int argc, char **argv) {
    if (argc != 4) {
        cout << "Usage: ../seg input_image initialization_file output_mask" << endl;
        return -1;
    }

    // Load the input image
    // the image should be a 3 channel image by default but we will double check that in teh seam_carving
    Mat in_image;
    in_image = imread(argv[1]/*, CV_LOAD_IMAGE_COLOR*/);

    if (!in_image.data) {
        cout << "Could not load input image!!!" << endl;
        return -1;
    }

    if (in_image.channels() != 3) {
        cout << "Image does not have 3 channels!!! " << in_image.depth() << endl;
        return -1;
    }

    // the output image
    Mat out_image = in_image.clone();

    ifstream f(argv[2]);
    if (!f) {
        cout << "Could not load initial mask file!!!" << endl;
        return -1;
    }

    int width = in_image.cols;
    int height = in_image.rows;


    //get the config pixels & save in array
    int n;
    f >> n;
    vector<Vertex> source_set;
    vector<Vertex> sink_set;

    for (int i = 0; i < n; ++i) {
        int x, y, t;
        f >> x >> y >> t;

        if (x < 0 || x >= width || y < 0 || y >= height) {
            cout << "I valid pixel mask!" << endl;
            return -1;
        }

        Vertex node = {y, x};

        if (t == 1) {
            source_set.push_back(node);
        } else {
            sink_set.push_back(node);
        }
    }

    //build adjacent matrix
    int num_vertex = width * height + 2;  // super source & super sink
    vector<vector<Edge> > adj_list(num_vertex);

    //build graph
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            int index = indexInMatrix(r, c, width);
            Vec3b the_pixel = out_image.at<Vec3b>(r, c);

            if (r != 0) {     //first row has not up vertex
                int up_index = indexInMatrix(r - 1, c, width);
                Vec3b up_pixel = out_image.at<Vec3b>(r - 1, c);
                int weight = weightBetweenPixel(the_pixel, up_pixel);
                Edge e = {index, up_index, weight};
                adj_list[index].push_back(e);
            }

            if (r != height - 1) {     //last row has not down vertex
                int down_index = indexInMatrix(r + 1, c, width);
                Vec3b down_pixel = out_image.at<Vec3b>(r + 1, c);
                int weight = weightBetweenPixel(the_pixel, down_pixel);
                Edge e = {index, down_index, weight};
                adj_list[index].push_back(e);
            }

            if (c != 0) {     //first col has not left vertex
                int left_index = indexInMatrix(r, c - 1, width);
                Vec3b left_pixel = out_image.at<Vec3b>(r, c - 1);
                int weight = weightBetweenPixel(the_pixel, left_pixel);
                Edge e = {index, left_index, weight};
                adj_list[index].push_back(e);
            }

            if (c != width - 1) {        //last col has not right vertex
                int right_index = indexInMatrix(r, c + 1, width);
                Vec3b right_pixel = out_image.at<Vec3b>(r, c + 1);
                int weight = weightBetweenPixel(the_pixel, right_pixel);
                Edge e = {index, right_index, weight};
                adj_list[index].push_back(e);
            }

        }
    }

    //connect super src with src_set
    int index_super_src = width * height;

    for (int i = 0; i < source_set.size(); ++i) {
        int index_src = indexInMatrix(source_set[i].row, source_set[i].col, width);
        Edge e = {index_super_src, index_src, 10000};
        adj_list[index_super_src].push_back(e);

        Edge e1 = {index_src, index_super_src, 0};
        adj_list[index_src].push_back(e1);
    }

    //connect super sink with sink_set
    int index_super_sink = width * height + 1;

    for (int i = 0; i < sink_set.size(); ++i) {
        int index_sink = indexInMatrix(sink_set[i].row, sink_set[i].col, width);
        Edge e = {index_sink, index_super_sink, 10000};
        adj_list[index_sink].push_back(e);

        Edge e1 = {index_super_sink, index_sink, 0};
        adj_list[index_super_sink].push_back(e1);
    }


    //ford-fulkerson algo
    while (true) {
        vector<Edge> augment_path = find_augment_path_bfs(adj_list, index_super_src, index_super_sink);

        if (augment_path.size() == 0) {      // not exist augment path
            break;
        }

        //find min flow among path
        int min_flow = INT_MAX;
        for (int e = 0; e < augment_path.size(); ++e) {
            if (augment_path[e].flow_capability < min_flow) {
                min_flow = augment_path[e].flow_capability;
            }
        }

        //adjust the flow capability
        for (int e = 0; e < augment_path.size(); ++e) {
            int index_from = augment_path[e].index_from;
            int index_to = augment_path[e].index_to;

            vector<Edge> edges = adj_list[index_from];
            int idx = 0;
            for (int i = 0; i < edges.size(); ++i) {
                if (edges[i].index_to == index_to) {
                    idx = i;
                    break;
                }
            }

            adj_list[index_from][idx].flow_capability -= min_flow;

            //find the opposite edge
            vector<Edge> opposite_edges = adj_list[index_to];
            int idx_ops = 0;
            for (int i = 0; i < opposite_edges.size(); ++i) {
                if (opposite_edges[i].index_to == index_from) {
                    idx_ops = i;
                    break;
                }
            }

            adj_list[index_to][idx_ops].flow_capability += min_flow;
        }

    }

    //cut the graph
    //color whole pic to background
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            Vec3b pixel;
            pixel[0] = 255;
            pixel[1] = 0;
            pixel[2] = 0;
            out_image.at<Vec3b>(r, c) = pixel;
        }
    }

    //bfs to color foreground
    //to record relations
    int parent[adj_list.size()];
    for (int i = 0; i < adj_list.size(); ++i) {
        parent[i] = -1;
    }

    queue<int> iqueue;
    iqueue.push(index_super_src);
    while (!iqueue.empty()) {
        int top = iqueue.front();
        //color
        if (top != index_super_src) {
            Vertex v = castIndex2Vertex(top, width);
            Vec3b pixel;
            pixel[0] = 0;
            pixel[1] = 0;
            pixel[2] = 255;
            out_image.at<Vec3b>(v.row, v.col) = pixel;
        }
        iqueue.pop();

        //search nbs
        vector<Edge> edges = adj_list[top];
        for (int i = 0; i < edges.size(); ++i) {
            if (edges[i].flow_capability > 0 && parent[edges[i].index_to] == -1) {
                parent[edges[i].index_to] = top;
                iqueue.push(edges[i].index_to);
            }
        }
    }

    // write it on disk
    imwrite(argv[3], out_image);
    // also display them both
    namedWindow("Original image", WINDOW_AUTOSIZE);
    namedWindow("Show Marked Pixels", WINDOW_AUTOSIZE);
    imshow("Original image", in_image);
    imshow("Show Marked Pixels", out_image);
    waitKey(0);
    return 0;
}


int indexInMatrix(int row, int col, int row_len) {
    return row * row_len + col;
}

Vertex castIndex2Vertex(int index, int row_len) {
    int row = (int) index / row_len;
    int col = index % row_len;
    Vertex v = {row, col};
    return v;
}

int weightBetweenPixel(Vec3b p1, Vec3b p2) {
    int diff = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) + abs(p1[2] - p2[2]);
    if (diff > 0) {
        return 1;
    } else {
        return 10000;
    }
}


vector<Edge> find_augment_path_bfs(vector<vector<Edge> > adj_list, int src, int tar) {
    vector<Edge> augment_path;

    //to record relations
    int index_parent[adj_list.size()];
    for (int i = 0; i < adj_list.size(); ++i) {
        index_parent[i] = -1;
    }

    //bfs
    queue<int> queue;
    queue.push(src);

    while (!queue.empty()) {
        int index_top = queue.front();
        queue.pop();

        //search nbs
        vector<Edge> edges = adj_list[index_top];
        for (int i = 0; i < edges.size(); ++i) {
            if (edges[i].flow_capability > 0 && index_parent[edges[i].index_to] == -1) {
                index_parent[edges[i].index_to] = index_top;
                queue.push(edges[i].index_to);
            }
        }
    }

    if (index_parent[tar] == -1) {      //no augment path exist
        return augment_path;
    } else {
        vector<int> index_in_path;
        int parent = index_parent[tar];
        index_in_path.push_back(tar);
        while (parent != src) {
            index_in_path.push_back(parent);
            parent = index_parent[parent];
        }
        index_in_path.push_back(src);

        //use index to find the edges in path
        for (long i = index_in_path.size() - 1; i > 0; --i) {
            int index = index_in_path[i];
            int index_next = index_in_path[i - 1];
            vector<Edge> edges = adj_list[index];
            for (int e = 0; e < edges.size(); ++e) {
                if (edges[e].index_to == index_next) {
                    augment_path.push_back(edges[e]);
                    break;
                }
            }
        }

        return augment_path;
    }
}






