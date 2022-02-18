#ifndef E3096311_CEA5_475E_847E_7C8C54043D10
#define E3096311_CEA5_475E_847E_7C8C54043D10

#include <utility>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>

namespace cppm
{
    typedef unsigned long uint64;
    typedef uint64 size_t[3];

    static const unsigned int __MAX_THREADS = std::thread::hardware_concurrency() / 2;

    template <class Type>
    class Matrix
    {
    private:
        Type *_elems;
        size_t _size;
        std::vector<uint64> _segmentsMul;
        std::vector<uint64> _segmentsLine;

        static void _hadPart(uint64 i0, uint64 const iMax, Type *relm, Type const *elm)
        {
            for (; i0 < iMax; ++i0)
                relm[i0] *= elm[i0];
        }
        static void _addPtrApply(uint64 i0, uint64 const iMax, Type *relm, Type const *elm, Type (*apply)(Type const))
        {
            for (; i0 < iMax; i0++)
                relm[i0] = apply(relm[i0] + elm[i0]);
        }
        static void _mulPtr_n(uint64 i0, uint64 const iMax, uint64 const kMax, uint64 const jMax, uint64 const offset,
            Type *relm, Type const *oelm, Type const *elm)
        {
            for(; i0 < iMax; ++i0)
                for(uint64 k = 0; k < kMax; ++k)
                    for(uint64 j = 0; j < jMax; ++j)
                        relm[i0 * jMax + j] += elm[i0 * offset + k] * oelm[k * jMax + j];
        }
        static void _addPtr_n(uint64 i0, uint64 const iMax, Type *relm, Type const *elm)
        {
            for (; i0 < iMax; ++i0)
                relm[i0] += elm[i0];
        }
        static void _subPtr_n(uint64 i0, uint64 const iMax, Type *relm, Type const *elm)
        {
            for (; i0 < iMax; ++i0)
                relm[i0] -= elm[i0];
        }
        template <class T2>
        static void _mulPtr_n2(uint64 i0, uint64 const iMax, Type *relm, T2 const elm)
        {
            for (; i0 < iMax; ++i0)
                relm[i0] *= elm;
        }
        template <class T2>
        static void _divPtr_n(uint64 i0, uint64 const iMax, Type *relm, T2 const elm)
        {
            for (; i0 < iMax; ++i0)
                relm[i0] /= elm;
        }
        static void memcpyType(Type *dest, Type const *source, uint64 i0, uint64 const iMax)
        {
            for (; i0 < iMax; ++i0)
                dest[i0] = source[i0];
        }
        static void _compN(Type const *a, Type const *b, uint64 i0, uint64 const iMax, bool *isEq)
        {
            for (; i0 < iMax && *isEq; i0++)
                if (a[i0] != b[i0]) {
                    *isEq = false;
                    break;
                }
        }
        static void _transposeN(Type *a, Type const *b, uint64 j, uint64 const jMax,
                                uint64 const oldCol, uint64 const oldRow)
        {
            for (; j < jMax; ++j)
                for (uint64 i = 0; i < oldCol; ++i)
                    a[i * oldRow + j] = b[j * oldCol + i];
        }
        void _initSegmentWith(std::vector<uint64> &s, uint64 const& max)
        {
            if (__MAX_THREADS == 0)
                return;

            uint64 start = 0;
            uint64 end = max / __MAX_THREADS;
            uint64 const offset = end;
            unsigned int i = 0;
            unsigned const n = (__MAX_THREADS - 1) * 2;

            for (; i < n; i += 2) {
                s.push_back(start);
                s.push_back(end);
                start += offset;
                end += offset;
            }
            s[i] = start;
            s[i + 1] = max;
        }
        void _initSegments(void)
        {
            _initSegmentWith(_segmentsMul, _size[0]);
            _initSegmentWith(_segmentsLine, _size[2]);
        }
        void _copyPtr(Matrix<Type> const *other)
        {
            _size[2] = other->_size[2];
            uint64 const n = _size[2];

            _size[0] = other->_size[0];
            _size[1] = other->_size[1];


            _initSegments();

            if (n >= __MAX_THREADS) {
                std::thread threads[__MAX_THREADS];

                for (int i = 0; i < __MAX_THREADS; i++)
                    threads[i] = std::thread(memcpyType, _elems, other->_elems,
                                            _segmentsLine[2 * i], _segmentsLine[2 * i + 1]);
                for (int i = 0; i < __MAX_THREADS; i++)
                    threads[i].join();
            } else
                memcpyType(_elems, other->_elems, 0, n);
        }
        void _addPtr(Matrix<Type> *cur, Matrix<Type> const& other) const
        {
            if (other._size[0] != cur->_size[0] || other._size[1] != cur->_size[1])
                throw "Incompatible sizes";

            uint64 const n = cur->_size[2];
            Type const *elm = other._elems;
            Type *relm = cur->_elems;

            if (n >= __MAX_THREADS) {
                std::thread threads[__MAX_THREADS];

                for (int i = 0; i < __MAX_THREADS; ++i)
                    threads[i] = std::thread(_addPtr_n,
                                             cur->_segmentsLine[2 * i], cur->_segmentsLine[2 * i + 1],
                                             relm, elm);
                for (int i = 0; i < __MAX_THREADS; ++i)
                    threads[i].join();

            } else {
                for (uint64 i = 0; i < n; ++i)
                    relm[i] += elm[i];
            }
        }
        void _minusPtr(Matrix<Type> *cur, Matrix<Type> const& other) const
        {
            if (other._size[0] != cur->_size[0] || other._size[1] != cur->_size[1])
                throw "Incompatible sizes";

            uint64 const n = cur->_size[2];
            Type const *elm = other._elems;
            Type *relm = cur->_elems;

            if (n >= __MAX_THREADS) {
                std::thread threads[__MAX_THREADS];

                for (int i = 0; i < __MAX_THREADS; ++i)
                    threads[i] = std::thread(_subPtr_n,
                                             cur->_segmentsLine[2 * i], cur->_segmentsLine[2 * i + 1],
                                             relm, elm);
                for (int i = 0; i < __MAX_THREADS; ++i)
                    threads[i].join();

            } else {
                for (uint64 i = 0; i < n; ++i)
                    relm[i] -= elm[i];
            }
        }
        Matrix<Type> _mulPtr(Matrix<Type> const& other) const
        {
            if (_size[1] != other._size[0])
                throw "Incompatible sizes";

            uint64 const kMax = other._size[0];
            uint64 const iMax = _size[0];
            uint64 const offset = _size[1];
            uint64 const jMax = other._size[1];
            Matrix<Type> result(iMax, jMax, true);

            Type *relm = result._elems;
            Type const *oelm = other._elems;
            Type const *elm = _elems;

            if (iMax >= __MAX_THREADS) {
                std::thread threads[__MAX_THREADS];

                for (unsigned i = 0; i < __MAX_THREADS; ++i)
                    threads[i] = std::thread(_mulPtr_n, _segmentsMul[2 * i], _segmentsMul[2 * i + 1], kMax, jMax, offset, relm, oelm, elm);
                for (unsigned i = 0; i < __MAX_THREADS; ++i)
                    threads[i].join();

            } else {
                for(uint64 i = 0; i < iMax; ++i)
                    for(uint64 k = 0; k < kMax; ++k)
                        for(uint64 j = 0; j < jMax; ++j)
                            relm[i * jMax + j] +=
                                elm[i * offset + k] * oelm[k * jMax + j];
            }
            return result;
        }

        template <class T2>
        void _mulConst(Matrix<Type> *cur, T2 const &other) const
        {
            uint64 const n = cur->_size[2];
            Type *elm = cur->_elems;

            if (n >= __MAX_THREADS) {
                std::thread threads[__MAX_THREADS];

                for (int i = 0; i < __MAX_THREADS; ++i)
                    threads[i] = std::thread(_mulPtr_n2<T2>,
                                            cur->_segmentsLine[2 * i],cur-> _segmentsLine[2 * i + 1],
                                            elm, other);
                for (int i = 0; i < __MAX_THREADS; ++i)
                    threads[i].join();

            } else {
                for (uint64 i = 0; i < n; ++i)
                    elm[i] *= other;
            }
        }

        template <class T2>
        void _divConst(Matrix<Type> *cur, T2 const &other) const
        {
            uint64 const n = cur->_size[2];
            Type *elm = cur->_elems;

            if (n >= __MAX_THREADS) {
                std::thread threads[__MAX_THREADS];

                for (int i = 0; i < __MAX_THREADS; ++i)
                    threads[i] = std::thread(_divPtr_n<T2>,
                                            cur->_segmentsLine[2 * i],cur-> _segmentsLine[2 * i + 1],
                                            elm, other);
                for (int i = 0; i < __MAX_THREADS; ++i)
                    threads[i].join();

            } else {
                for (uint64 i = 0; i < n; ++i)
                    elm[i] /= other;
            }
        }
    public:
        Matrix(uint64 const nb_line = 1, uint64 const nb_col = 1, bool const fill = false, Type const filler = Type())
        {
            if (!nb_line || !nb_col)
                throw "Invalid null size";
            _size[0] = nb_line;
            _size[1] = nb_col;
            _size[2] = nb_col * nb_line;

            _initSegments();

            _elems = new Type[_size[2]];
            if (fill)
                for (uint64 i = 0; i < _size[2]; ++i)
                    _elems[i] = filler;
        }
        Matrix(Matrix<Type> const& other)
        {
            _elems = new Type[other._size[2]];
            _copyPtr(&other);
        }
        Matrix(Matrix<Type> const *other)
        {
            _elems = new Type[other->_size[2]];
            _copyPtr(other);
        }
        ~Matrix()
        {
            delete [] _elems;
        }

        Matrix<Type>& operator=(Matrix<Type> const& other)
        {
            if (_size[2] != other._size[2]) {
                delete [] _elems;
                _elems = new Type[other._size[2]];
            }
            _copyPtr(&other);
            return *this;
        }

        Matrix<Type> operator+(Matrix<Type> const &other) const
        {
            Matrix<Type> r(this);
            _addPtr(&r, other);
            return r;
        }

        Matrix<Type> operator-(Matrix<Type> const &other) const
        {
            Matrix<Type> r(this);
            _minusPtr(&r, other);
            return r;
        }

        Matrix<Type> operator*(Matrix<Type> const &other) const
        {
            return _mulPtr(other);
        }

        template<class T2>
        Matrix<Type> operator*(T2 const& other) const
        {
            Matrix<Type> r(this);
            _mulConst<T2>(&r, other);
            return r;
        }
        template<class T2>
        Matrix<Type> operator/(T2 const& other) const
        {
            Matrix<Type> r(this);
            _divConst<T2>(&r, other);
            return r;
        }

        template <class T2>
        friend Matrix<Type> operator*(T2 const &other, Matrix<Type> const& a)
        {
            Matrix<Type> r(a);
            r._mulConst<T2>(&r, other);
            return r;
        }

        bool operator==(Matrix<Type> const& other)
        {
            if (_size[0] != other._size[0] || _size[1] != other._size[1])
                return false;
            bool isEq = true;

            uint64 const n = _size[2];
            Type *elm = other._elems;

            if (n >= __MAX_THREADS) {
                std::thread threads[__MAX_THREADS];

                for (int i = 1; i < __MAX_THREADS; ++i)
                    threads[i] = std::thread(_compN, _elems, elm, _segmentsLine[2 * i], _segmentsLine[2 * i + 1], &isEq);
                for (int i = 1; i < __MAX_THREADS; ++i)
                    threads[i].join();

            } else
                _compN(_elems, elm, 0, n, &isEq);
            return isEq;
        }
        bool operator!=(Matrix<Type> const& other)
        {
            return !operator==(other);
        }

        static Matrix<Type> identity(uint64 const& size, Type const& unity)
        {
            Matrix<Type> r(size, size, true);

            for (uint64 i = 0; i < size; i++)
                r._elems[i * size + i] = unity;
            return r;
        }

        Matrix<Type> &operator+=(Matrix<Type> const& other)
        {
            _addPtr(this, other);
            return *this;
        }
        Matrix<Type> &operator-=(Matrix<Type> const& other)
        {
            _minusPtr(this, other);
            return *this;
        }

        Matrix<Type> &transpose(void)
        {
            Matrix<Type> copy(this);

            uint64 const oldRow = _size[0];
            uint64 const oldCol = _size[1];
            _size[0] = oldCol;
            _size[1] = oldRow;
            Type const *oelm = copy._elems;

            if (oldCol >= __MAX_THREADS) {
                std::thread threads[__MAX_THREADS];

                for (int i = 0; i < __MAX_THREADS; ++i)
                    threads[i] = std::thread(_transposeN,
                                             _elems, oelm,
                                             _segmentsMul[2 * i], _segmentsMul[2 * i + 1],
                                             oldCol, oldRow);
                for (int i = 0; i < __MAX_THREADS; ++i)
                    threads[i].join();

            } else {
                for (uint64 j = 0; j < oldRow; ++j)
                    for (uint64 i = 0; i < oldCol; ++i)
                        _elems[i * oldRow + j] = oelm[j * oldCol + i];
            }
            return *this;
        }
        Matrix<Type> getTranspose(void)
        {
            Matrix<Type> cpy(this);
            cpy.transpose();
            return cpy;
        }
        Matrix<Type> &operator*=(Matrix<Type> const& other)
        {
            Matrix<Type> cpy = _mulPtr(other);

            return operator=(cpy);
        }
        template<class T2>
        Matrix<Type> &operator*=(T2 const& other)
        {
            _mulConst<T2>(this, other);
            return *this;
        }
        template<class T2>
        Matrix<Type> &operator/=(T2 const& other)
        {
            _divConst<T2>(this, other);
            return *this;
        }

        Matrix<Type> addAndApply(Matrix<Type> const& other, Type (*apply)(Type const))
        {
            if (other._size[0] != _size[0] || other._size[1] != _size[1])
                throw "Incompatible sizes";
            Matrix<Type> r(this);

            uint64 const n = _size[2];
            Type const *elm = other._elems;
            Type *relm = r._elems;

            if (n >= __MAX_THREADS) {
                std::thread threads[__MAX_THREADS];

                for (int i = 0; i < __MAX_THREADS; ++i)
                    threads[i] = std::thread(_addPtrApply,
                                             _segmentsLine[2 * i], _segmentsLine[2 * i + 1],
                                             relm, elm, apply);
                for (int i = 0; i < __MAX_THREADS; ++i)
                    threads[i].join();

            } else {
                for (uint64 i = 0; i < n; ++i)
                    r._elems[i] = apply(r._elems[i] + other._elems[i]);
            }
            return r;
        }
        Matrix<Type> hadamard(Matrix<Type> const& other)
        {
            if (other._size[0] != _size[0] || other._size[1] != _size[1])
                throw "Incompatible sizes in hadamard product";
            Matrix<Type> r(this);

            uint64 const n = _size[2];
            Type const *elm = other._elems;
            Type *relm = r._elems;

            if (n >= __MAX_THREADS) {
                std::thread threads[__MAX_THREADS];

                for (int i = 0; i < __MAX_THREADS; ++i)
                    threads[i] = std::thread(_hadPart,
                                             _segmentsLine[2 * i], _segmentsLine[2 * i + 1],
                                             relm, elm);
                for (int i = 0; i < __MAX_THREADS; ++i)
                    threads[i].join();

            } else {
                for (uint64 i = 0; i < n; i++)
                    relm[i] *= elm[i];
            }
            return r;
        }
        const size_t& getSize() const {return _size;}
        Type &at(uint64 const i, uint64 const j) const
        {
            if (i >= _size[0])
                throw "Bad row index: " + std::to_string(i);
            if (j >= _size[1])
                throw "Bad column index: " + std::to_string(j);
            return _elems[i * _size[1] + j];
        }
        const Type *getElems(void) const {return _elems;}
    };
}

#endif // E3096311_CEA5_475E_847E_7C8C54043D10
