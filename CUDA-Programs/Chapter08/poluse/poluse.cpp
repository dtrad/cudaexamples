// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// chapter 8 poluse
// This is a support program for chapter 8 PET simualtion
// It uses the lookup table generated by polmake to convert
// output of the reco program from a polar to a cartesian 
// coordinate grid. The Cartesian voxels have dimesions equal to 
// the ring spacing. Thus a 200 x 200  x-y grid is used for 100 rings.

// C:\c:\Users\Richard\OneDrive\toGit2\bin\poluse.exe reco_mlem020.raw pol2cart.tab cart_mlem050.raw
// file reco_mlem020.raw read
// file pol2cart.tab read
// file cart_mlem050.raw written

#include "cx.h"  // host only cx
#include "cxbinio.h"
#include "scanner.h"

#include <vector>

struct cp_grid_map {
	float b[voxBox][voxBox];
	int x; // carteisian origin
	int y;
	int phi;  // polar voxel
	int r;
};

int main(int argc,char *argv[])
{
	if(argc < 2) {
		printf("usage poluse.exe <input polar file> <polmap> <output cartesian file>\n");
		return 0;	
	}

	int pol_size =  cryNum*zNum*radNum;  // NB order [ring, z, phi]
	int cart_size = voxNum*voxNum*zNum;  //          [2*z,    y,   x]
	int map_size =  cryNum*radNum;       //          [ring, phi]
	
	std::vector<float>        pol(pol_size);
	std::vector<float>       cart(cart_size);
	std::vector<cp_grid_map>  map(map_size);

	if(cx::read_raw(argv[1],pol.data(),pol_size)){printf("bad read on %s\n",argv[1]); return 1;}
	if(cx::read_raw(argv[2],map.data(),map_size)){printf("bad read on %s\n",argv[2]); return 1;}

	for(int r=0;r<radNum;r++) for(int z=0;z<zNum;z++) for(int p=0;p<cryNum;p++){
		float val = pol[(r*zNum+z)*cryNum+p];

		float vol_fraction =  1.0f;  //2*r+1;
		int index = r*cryNum+p;
		if(val > 0.0f){
			int x0 = map[index].x;
			int y0 = map[index].y;
			for(int i=0;i<voxBox;i++) {
				int y = y0+i;
				if(y>=0 && y<voxNum) for(int j= 0;j<voxBox;j++){
					int x = x0+j;
					if(x>=0 && x <voxNum && map[index].b[i][j]>0.0f) cart[(z*voxNum+y)*voxNum+x] += vol_fraction*val*map[index].b[i][j];
				}
			}
		}
	}

	cx::write_raw(argv[3],cart.data(),cart_size);

	return 0;
}


