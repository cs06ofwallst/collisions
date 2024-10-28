#include <stdio.h>
#include <cuda_runtime.h>
#include "collisions_continuous.cu"
__device__ float Side(float3 a, float3 b, float3 c, float3 l) {
    // (a - c) and (b - c)
    float3 ac = make_float3(a.x - c.x, a.y - c.y, a.z - c.z);
    float3 bc = make_float3(b.x - c.x, b.y - c.y, b.z - c.z);

    // cross product (a - c) x (b - c)
    float3 cross_product = make_float3(
        ac.y * bc.z - ac.z * bc.y,
        ac.z * bc.x - ac.x * bc.z,
        ac.x * bc.y - ac.y * bc.x
    );

    float side = cross_product.x * l.x + cross_product.y * l.y + cross_product.z * l.z;
    return side;
}

__device__ int SideSign(float3 a, float3 b, float3 c, float3 l) {
    float side = Side(a, b, c, l);
    if (side > 0) {
        return 1;
    } else if (side < 0) {
        return -1;
    } else {
        return 0;
    }
}
__global__ bool UnprojectedContourTest(float3* vertices, float3 coneAxis) {
    int numVertices = sizeof(vertices)/sizeof(float3);
    int i = threadIdx.x;
    if (i >= numVertices - 1) return;

    // init intersection nr to 0
    __shared__ int intNum;
    if (threadIdx.x == 0) intNum = 0;
    __syncthreads();

    // determine self intersection
    __shared__ int res;
    if (threadIdx.x == 0) res = 1;  // assume true
    __syncthreads();

    // params
    float3 o = make_float3(0, 0, 0);
    for (int j = 0; j < numVertices; j++) {
        o.x += vertices[j].x;
        o.y += vertices[j].y;
        o.z += vertices[j].z;
    }
    o.x /= numVertices;
    o.y /= numVertices;
    o.z /= numVertices;

    // r is the axis perpendicular to the cone axis
    float3 r = make_float3(1, 0, 0);
    //check parallel to (0,1,0)
    if (coneAxis.x == 0.0f && coneAxis.z == 0.0f && coneAxis.y != 0.0f) {
        float3 r = make_float3(1, 0, 0);
    } else {
        float3 r1 = make_float3(0, 1, 0);
        float3 r = make_float3(0, coneAxis.y, 0);
    }
    // side sign at 1st contour segment
    int s0 = SideSign(o, vertices[0], vertices[1], coneAxis);  // TODO: implement
    if (s0 == 0) {
        return false;
    }
    __syncthreads();
   int n = 0;
   for (int i=0; i<numVertices; i++) {
    
    // side + intersection test on each contour segment in parallel
        // side
        int s1 = SideSign(o, vertices[i], vertices[(i + 1) % numVertices], coneAxis); 
        if (s0 != s1) {
            return false;  
        }

        // intersection
        int s2 = SideSign(vertices[i], o, o + r, coneAxis);  
        int s3 = SideSign(vertices[i+1], o, o + r, coneAxis);
        if (s3 == 0 || s2 == 0) {
            return false;  
        }

        if (s3 == s0 && s3 != s2) {
            n++;  // atomic operation for shared variable
            if (n > 1) {
                return false;
            }
        }
    
     __syncthreads();
   }
  return true;
}

// example main
int main() {
    int numVertices = 4;

    // cone axis and vertices, does it work like this?
    float3 coneAxis = make_float3(0, 0, 1);
    float3 vertices[] = {
        make_float3(1, 0, 0),
        make_float3(0, 1, 0),
        make_float3(-1, 0, 0),
        make_float3(0, -1, 0)
    };

    // device
    float3* d_vertices;
    int* d_result;
    int result = 0;

    // alloc mem
    cudaMalloc((void**)&d_vertices, numVertices * sizeof(float3));
    cudaMalloc((void**)&d_result, sizeof(int));

    // copy to device
    cudaMemcpy(d_vertices, vertices, numVertices * sizeof(float3), cudaMemcpyHostToDevice);

    // kernel
    UnprojectedContourTest<<<1, numVertices>>>(d_vertices, numVertices, coneAxis, d_result);

    // back to host
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(d_vertices);
    cudaFree(d_result);

    // output
    if (result == 1) {
        printf("No self-intersection on the projected contour.\n");
    } else {
        printf("Self-intersection detected on the projected contour.\n");
    }

    return 0;
}

// BVH Node structure
struct BVHNode {
    bool isLeaf;         // leaf?
    float apexAngle;     // apex andgle of normal cone
    float3* vertices;    // vertices on node
    int numVertices;     // nr of vertices
    BVHNode* leftChild;  // reference
    BVHNode* rightChild; // reference
    float3 coneAxis;     // normal cone axis
};

// idk
__device__ bool UnprojectedContourTest(float3* vertices, int numVertices, float3 coneAxis, float apexAngle);
__device__ void Collide(BVHNode* leftNode, BVHNode* rightNode);

// recursive function
__device__ void SelfCollide(BVHNode* N) {
    // base (leaf -> terminate)
    if (N->isLeaf) {
        return;  // Traversal terminated
    }

    // is the apex angle of the normal cone less than pi
    if (N->apexAngle < M_PI) {
        // unprojected contour test
        if (UnprojectedContourTest(N->vertices, N->numVertices, N->coneAxis, N->apexAngle)) {
            return;  // no self-collisions
        }
    }

    // check left and right children
    SelfCollide(N->leftChild);
    SelfCollide(N->rightChild);

    // collisions between left and right children
    Collide(N->leftChild, N->rightChild);
}

// handle collisions between two BVH nodes
__device__ void Collide(BVHNode* leftNode, BVHNode* rightNode) {
    // TODO
}

// init SelfCollide on a BVH root
__global__ void SelfCollideKernel(BVHNode* root) {
    // Launch the SelfCollide recursive function on the BVH root
    SelfCollide(root);
}

int main() {
    // example
    BVHNode root;
    BVHNode leftChild, rightChild;

    // build
    root.isLeaf = false;
    root.apexAngle = M_PI / 4;  
    root.coneAxis = make_float3(0, 0, 1);  
    root.leftChild = &leftChild;
    root.rightChild = &rightChild;

    leftChild.isLeaf = true;  // left is leaf
    rightChild.isLeaf = true;  // right is leaf

    //mem alloc
    BVHNode* d_root;
    cudaMalloc((void**)&d_root, sizeof(BVHNode));

    // copy to device
    cudaMemcpy(d_root, &root, sizeof(BVHNode), cudaMemcpyHostToDevice);

    // 1 thread (recursive)
    SelfCollideKernel<<<1, 1>>>(d_root);

    // synchr
    cudaDeviceSynchronize();

    // free
    cudaFree(d_root);

    return 0;
}

__device__ int CSideSign1(float3 o, float3 v1, float3 v2, float3 l) {
    // TODO
    return 0; 
}
__device__ int CSideSign2(float3 o, float3 v1, float3 v2, float3 l) {
    // TODO
    return 0; 
}
__global__ void UnprojectedContourTestForCCD(float3* vertices, float3 coneAxis) {
    int numVertices = sizeof(vertices)/sizeof(float3);
    int i = threadIdx.x;
    if (i >= numVertices - 1) return;

    // init intersection nr to 0
    __shared__ int intNum;
    if (threadIdx.x == 0) intNum = 0;
    __syncthreads();

    // determine self intersection
    __shared__ int res;
    if (threadIdx.x == 0) res = 1;  // assume true
    __syncthreads();

    // params
    float3 o = make_float3(0, 0, 0);
    for (int j = 0; j < numVertices; j++) {
        o.x += vertices[j].x;
        o.y += vertices[j].y;
        o.z += vertices[j].z;
    }
    o.x /= numVertices;
    o.y /= numVertices;
    o.z /= numVertices;

    // r is the axis perpendicular to the cone axis
    float3 r = make_float3(1, 0, 0);
    //check parallel to (0,1,0)
    if (coneAxis.x == 0.0f && coneAxis.z == 0.0f && coneAxis.y != 0.0f) {
        float3 r = make_float3(1, 0, 0);
    } else {
        float3 r1 = make_float3(0, 1, 0);
        float3 r = make_float3(0, coneAxis.y, 0);
    }
    // side sign at 1st contour segment
    int s0 = CSideSign1(o, vertices[0], vertices[1], coneAxis);  // TODO: implement
    if (s0 == 0) {
        return false;
    }
    __syncthreads();
   int n = 0;
   for (int i=0; i<numVertices; i++) {
    
    // side + intersection test on each contour segment in parallel
        // side
        int s1 = CSideSign1(o, vertices[i], vertices[(i + 1) % numVertices], coneAxis); 
        if (s0 != s1) {
            return false;  
        }

        // intersection
        int s2 = CSideSign2(vertices[i], o, o + r, coneAxis);  
        int s3 = CSideSign2(vertices[i+1], o, o + r, coneAxis);
        if (s3 == 0 || s2 == 0) {
            return false;  
        }

        if (s3 == s0 && s3 != s2) {
            n++;  // atomic operation for shared variable
            if (n > 1) {
                return false;
            }
        }
    
     __syncthreads();
   }
  return true;
}

int main() {
    // jsut another example
    int numVertices = 5;
    float alpha = 45.0f;  // apex angle
    float3 l = make_float3(0, 1, 0);  // axis of the cone

    float3 h_vertices[] = {
        make_float3(1, 0, 0),
        make_float3(0, 1, 0),
        make_float3(-1, 0, 0),
        make_float3(0, -1, 0),
        make_float3(0.5f, 0.5f, 0)
    };

    float3* d_vertices;
    bool* d_result;
    bool h_result;

    cudaMalloc(&d_vertices, numVertices * sizeof(float3));
    cudaMalloc(&d_result, sizeof(bool));

    // to device
    cudaMemcpy(d_vertices, h_vertices, numVertices * sizeof(float3), cudaMemcpyHostToDevice);

    // kernel
    UnprojectedContourTestForCCD<<<1, 1>>>(d_vertices, numVertices, alpha, l, d_result);

    // back
    cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);

    // check
    if (h_result) {
        printf("No self-intersection on the projected contour.\n");
    } else {
        printf("Self-intersection detected or result is undetermined.\n");
    }

    cudaFree(d_vertices);
    cudaFree(d_result);

    return 0;
}

struct NormalCone {
    float apexAngle;
    // ...
};

struct BVTTFront {
};

__device__ bool IsLeaf(const BVHNode* node) {
    // TODO: deltermine if it's a leaf
    return node->isLeaf;
}

__device__ void FrontTracking(const BVTTFront* frontN) {
    // TODO: impl front tracking
}

__device__ void SelfCollideWithGuidedFrontTracking(BVHNode* N, BVTTFront* FrontN, NormalCone* CN) {
    // leaf?
    if (IsLeaf(N)) {
        return;  // done
    }

    // apex angle
    if (CN->apexAngle < M_PI) {
        // unprojected contour test
        if (UnprojectedContourTest(CN)) {
            return;  // free
        }
    }

    // recurse
    SelfCollideWithGuidedFrontTracking(N->leftChild, FrontN, CN);
    SelfCollideWithGuidedFrontTracking(N->rightChild, FrontN, CN);

    Collide(N->leftChild, N->rightChild);
}
