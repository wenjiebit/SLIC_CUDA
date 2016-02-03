//
// Created by derue on 15/12/15.
//

#include <driver_types.h>
#include "SLIC_cuda.h"

using namespace std;


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


SLIC_cuda::SLIC_cuda(int diamSpx, float wc){
    m_diamSpx = diamSpx;
    m_wc = wc;
}
SLIC_cuda::~SLIC_cuda(){
    delete[] m_clusters;
    delete[] m_labels;
}
void SLIC_cuda::Initialize(cv::Mat &frame0) {
    m_width = frame0.cols;
    m_height = frame0.rows;
    m_nPx = m_width*m_height;
    getWlHl(m_width, m_height, m_diamSpx, m_wSpx, m_hSpx); // determine w and h of Spx based on diamSpx
    m_areaSpx = m_wSpx*m_hSpx;
    CV_Assert(m_nPx%m_areaSpx==0);
    m_nSpx = m_nPx/m_areaSpx; 
    m_clusters = new float[m_nSpx * 5];
    m_labels = new float[m_nPx];
    InitBuffers();
}
void SLIC_cuda::Segment(cv::Mat &frame) {
	m_nSpx = m_nPx / m_areaSpx;// should be an integer!!(needed here to reinint because modified in enforce connect)
    SendFrame(frame); //ok
    InitClusters();//ok
    for(int i=0; i<N_ITER; i++) {
        Assignement();
        //cudaDeviceSynchronize();
        Update();
        //cudaDeviceSynchronize();
    }
	Assignement();
	//CPU stuff like SIFT computation

	//GetLabels();
	//EnforceConnectivity();
}

void SLIC_cuda::InitBuffers() {

    //allocate buffers on gpu
    cudaChannelFormatDesc channelDescr = cudaCreateChannelDesc(8,8,8,8,cudaChannelFormatKindUnsigned);
    gpuErrchk(cudaMallocArray(&frameBGRA_array,&channelDescr,m_width,m_height));

	cudaChannelFormatDesc channelDescrLab = cudaCreateChannelDesc(8,8,8,8,cudaChannelFormatKindUnsigned);
    gpuErrchk(cudaMallocArray(&frameLab_array,&channelDescrLab,m_width,m_height,cudaArraySurfaceLoadStore));

    cudaChannelFormatDesc channelDescrLabels = cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat);
    gpuErrchk(cudaMallocArray(&labels_array,&channelDescrLabels,m_width,m_height,cudaArraySurfaceLoadStore));


    //texture FrameBGR (read-only)
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = frameBGRA_array;

    // Specify texture object parameters
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0]    = cudaAddressModeClamp;
    texDesc.addressMode[1]    = cudaAddressModeClamp;
    texDesc.filterMode       = cudaFilterModePoint;
    texDesc.readMode         = cudaReadModeElementType;
    texDesc.normalizedCoords = false;

    gpuErrchk(cudaCreateTextureObject(&frameBGRA_tex, &resDesc,&texDesc,NULL));


    // surface frameLab
    cudaResourceDesc resDescLab;
    memset(&resDescLab, 0, sizeof(resDescLab));
    resDescLab.resType = cudaResourceTypeArray;

    resDescLab.res.array.array = frameLab_array;
    gpuErrchk(cudaCreateSurfaceObject(&frameLab_surf, &resDescLab));

    // surface labels
    cudaResourceDesc resDescLabels;
    memset(&resDescLabels, 0, sizeof(resDescLabels));
    resDescLabels.resType = cudaResourceTypeArray;

    resDescLabels.res.array.array = labels_array;
    gpuErrchk(cudaCreateSurfaceObject(&labels_surf, &resDescLabels));

    // buffers clusters , accAtt
    gpuErrchk(cudaMalloc((void**)&clusters_g, m_nSpx*sizeof(float)*5)); // 5-D centroid
    gpuErrchk(cudaMalloc((void**)&accAtt_g, m_nSpx*sizeof(float)*6)); // 5-D centroid acc + 1 counter
    cudaMemset(accAtt_g, 0, m_nSpx*sizeof(float)*6);//initialize accAtt to 0

	//new buffer to avoid atomic
	float total_pixel_to_search = (float)(m_diamSpx * m_diamSpx * 9);
	int nBlock_per_spx = (int)ceil(total_pixel_to_search / (float)(BLOCK_DIM * BLOCK_DIM));
	gpuErrchk(cudaMalloc((void**)&accum_map, m_nSpx*sizeof(float) * 6 * nBlock_per_spx));

}



void SLIC_cuda::SendFrame(cv::Mat& frameBGR){

    cv::Mat frameBGRA;
    cv::cvtColor(frameBGR,frameBGRA,CV_BGR2BGRA);
    CV_Assert(frameBGRA.type()==CV_8UC4);
    CV_Assert(frameBGRA.isContinuous());

    cudaMemcpyToArray(frameBGRA_array,0,0,(uchar*)frameBGRA.data,m_nPx* sizeof(uchar4),cudaMemcpyHostToDevice); //ok

    Rgb2CIELab(frameBGRA_tex,frameLab_surf,m_width,m_height); //BGR->Lab gpu

}

void SLIC_cuda::GetLabels()
{
	gpuErrchk(cudaMemcpyFromArray(m_labels, labels_array, 0, 0, m_nPx* sizeof(float), cudaMemcpyDeviceToHost));

	//cudaMemcpyFromArray()

}


void SLIC_cuda::EnforceConnectivity()
{
	int label = 0, adjlabel = 0;
	int lims = (m_width * m_height) / (m_nSpx);
	lims = lims >> 2;


	const int dx4[4] = { -1, 0, 1, 0 };
	const int dy4[4] = { 0, -1, 0, 1 };

	vector<vector<int> >newLabels;
	for (int i = 0; i < m_height; i++)
	{
		vector<int> nv(m_width, -1);
		newLabels.push_back(nv);
	}

	for (int i = 0; i < m_height; i++)
	{
		for (int j = 0; j < m_width; j++)
		{
			if (newLabels[i][j] == -1)
			{
				vector<cv::Point> elements;
				elements.push_back(cv::Point(j, i));
				for (int k = 0; k < 4; k++)
				{
					int x = elements[0].x + dx4[k], y = elements[0].y + dy4[k];
					if (x >= 0 && x < m_width && y >= 0 && y < m_height)
					{
						if (newLabels[y][x] >= 0)
						{
							adjlabel = newLabels[y][x];
						}
					}
				}
				int count = 1;
				for (int c = 0; c < count; c++)
				{
					for (int k = 0; k < 4; k++)
					{
						int x = elements[c].x + dx4[k], y = elements[c].y + dy4[k];
						if (x >= 0 && x < m_width && y >= 0 && y < m_height)
						{
							if (newLabels[y][x] == -1 && m_labels[i*m_width+j] == m_labels[y*m_width+x])
							{
								elements.push_back(cv::Point(x, y));
								newLabels[y][x] = label;//m_labels[i][j];
								count += 1;
							}
						}
					}
				}
				if (count <= lims) {
					for (int c = 0; c < count; c++) {
						newLabels[elements[c].y][elements[c].x] = adjlabel;
					}
					label -= 1;
				}
				label += 1;
			}
		}
	}
	m_nSpx = label;
	/*for (int i = 0; i < newLabels.size(); i++)
		for (int j = 0; j < newLabels[i].size(); j++)
			m_labels[i*m_width+j] = newLabels[i][j];*/

	cudaDeviceSynchronize();
}




void SLIC_cuda::displayBound(cv::Mat &image, cv::Scalar colour)
{
    //load label from gpu
	//GetLabels();

    const int dx8[8] = { -1, -1,  0,  1, 1, 1, 0, -1 };
    const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1 };

    /* Initialize the contour vector and the matrix detailing whether a pixel
    * is already taken to be a contour. */
    vector<cv::Point> contours;
    vector<vector<bool> > istaken;
    for (int i = 0; i < image.rows; i++) {
        vector<bool> nb;
        for (int j = 0; j < image.cols; j++) {
            nb.push_back(false);
        }
        istaken.push_back(nb);
    }

    /* Go through all the pixels. */

    for (int i = 0; i<image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {

            int nr_p = 0;

            /* Compare the pixel to its 8 neighbours. */
            for (int k = 0; k < 8; k++) {
                int x = j + dx8[k], y = i + dy8[k];

                if (x >= 0 && x < image.cols && y >= 0 && y < image.rows) {
                    if (istaken[y][x] == false && m_labels[i*m_width+j] != m_labels[y*m_width+x]) {
                        nr_p += 1;
                    }
                }
            }
            /* Add the pixel to the contour list if desired. */
            if (nr_p >= 2) {
                contours.push_back(cv::Point(j, i));
                istaken[i][j] = true;
            }

        }
    }

    /* Draw the contour pixels. */
    for (int i = 0; i < (int)contours.size(); i++) {
        image.at<cv::Vec3b>(contours[i].y, contours[i].x) = cv::Vec3b(colour[0], colour[1], colour[2]);
    }
}


