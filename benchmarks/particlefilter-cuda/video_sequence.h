#ifndef VIDEO_SEQUENCE_H_
#define VIDEO_SEQUENCE_H_
#define HOST_DEVICE __host__ __device__
class FrameInnerAccessPolicy {
public:
  template<typename Underlying>
  HOST_DEVICE static Underlying &access(int IsX, int IszY, int Nfr, Underlying *I, int x, int y, int z) {
    return I[get_index(IsX, IszY, Nfr, x, y, z)];
  }

  HOST_DEVICE static inline int get_index(int IsX, int IszY, int Nfr, int x, int y, int z) {
    return x * IszY * Nfr + y * Nfr + z;
  }
};

class FrameOuterAccessPolicy {
public:
  template<typename Underlying>
  HOST_DEVICE static Underlying &access(int IsX, int IszY, int Nfr, Underlying *I, int x, int y, int z) {
    return I[get_index(IsX, IszY, Nfr, x, y, z)];
  }

  HOST_DEVICE static inline int get_index(int IsX, int IszY, int Nfr, int x, int y, int z) {
    return z * IsX * IszY + y * IsX + x;
  }
};

template<typename AccessPolicy>
class VideoSequence {
public:
  int IszX;
  int IszY;
  int Nfr;
  int *seed;
  bool owns_I = true;
  unsigned char *I;

  VideoSequence(int IszX, int IszY, int Nfr, int * seed) : IszX(IszX), IszY(IszY), Nfr(Nfr), seed(seed) {
    owns_I = true;
    I = (unsigned char *) calloc(IszX * IszY * Nfr, sizeof(unsigned char));
  }

  VideoSequence(unsigned char* I, int IszX, int IszY, int Nfr, int * seed) : IszX(IszX), IszY(IszY), Nfr(Nfr), seed(seed), I(I) {
    owns_I = false;
  }

  ~VideoSequence() {
    if(owns_I)
      free(I);
  }

  HOST_DEVICE unsigned char &operator()(int x, int y, int frame) {
    return this->access(x, y, frame);
  }
  HOST_DEVICE unsigned char &operator()(size_t direct) {
    return this->access(direct);
  }

  HOST_DEVICE unsigned char &access(int x, int y, int frame) {
    return AccessPolicy::access(IszX, IszY, Nfr, I, x, y, frame);
  }

  HOST_DEVICE unsigned char &access(size_t index) {
    return I[index];
  }

  HOST_DEVICE int get_index(int x, int y, int frame) {
    return AccessPolicy::get_index(IszX, IszY, Nfr, x, y, frame);
  }

  HOST_DEVICE int xdim() const {
    return IszX;
  }
  HOST_DEVICE int ydim() const {
    return IszY;
  }
  HOST_DEVICE int zdim() const {
    return Nfr;
  }

  HOST_DEVICE int *seed_array() const {
    return seed;
  }

  HOST_DEVICE unsigned char *data() const {
    return I;
  }
};


#endif // VIDEO_SEQUENCE_H_