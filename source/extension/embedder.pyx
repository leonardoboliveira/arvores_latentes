import numpy as np
from libcpp cimport bool

cimport numpy
cimport cython

cdef divide(a, b):
    return np.divide(a, b)


cdef subtract(a, b):
    return np.subtract(a, b)


cdef class BertEmbedder:
    cdef double[:, :] embeddings
    cdef int num_cols

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.num_cols = embeddings.shape[1]

    def get_embedding(self, begin, end):
        if begin >= end:
            return np.zeros(self.num_cols)

        last_line = self.embeddings[end - 1, :]
        if begin == 0:
            embedding = divide(last_line, end)
        else:
            first_line = self.embeddings[begin - 1, :]
            embedding = divide(subtract(last_line, first_line) , (end - begin))

        return embedding


cdef class BinsEmbedder(BertEmbedder):
    cdef double[:, :] bins
    cdef double[:, :] eye

    def __init__(self, embeddings, bins, eye):
        BertEmbedder.__init__(self, embeddings)
        self.bins = bins
        self.eye = eye

    def get_embedding(self, begin, end, as_category=False, return_embedding=False):
        if begin >= end:
            return 0

        embedding = super().get_embedding(begin, end)

        if as_category:
            ret_bins = self.to_classes(embedding)
        else:
            ret_bins = self.to_bins(embedding)

        if return_embedding:
            return ret_bins, list(embedding)
        return ret_bins

    cdef long[:] to_classes(self, double[:] embedding):
        classes = [self.to_bin(idx, x) for idx, x in enumerate(embedding)]
        return np.array(classes)

    cdef double[:] to_bins(self, double[:] embedding):
        classes = self.to_classes(embedding)
        # return np.reshape(self.eye[classes], -1)
        eyes = [self.eye[x] for x in classes]
        return np.concatenate(eyes)

    cdef size_t to_bin(self, int pos, double x):
        for idx, value in enumerate(self.bins[:, pos]):
            if x < value:
                return idx
        return len(self.bins)


cdef class TemplateEmbedder:
    cdef long long[:, :] template

    def __init__(self, template):
        self.template = template

    cdef int derived_match(self, double[:] full, long long[:] basic_features):
        for i, idx in enumerate(basic_features):
            #print(f"Checking {i}:{idx}:{full[idx]}")
            if full[idx] == 0:
                return 0
        return 1

    def template_match(self, full):
        # print(f"Templates:{self.template.shape}")
        for i in range(self.template.shape[0]):
            basic_feature = self.template[i,:]
            # print(f"Matching {i}")
            if self.derived_match(full, basic_feature) == 1:
                # print(f"Match template {i}")
                return basic_feature
        return None

class InduceEmbedder:

    def __init__(self, derived_features):
        self.derived_features = [TemplateEmbedder(t.values) for t in derived_features]

    def get_features(self, full):
        features = []
        for template in self.derived_features:
            matched = template.template_match(full)
            if matched is not None:
                features.append(matched)

        # if len(features) > 0:
        #     print(f"Has something:{features}")
        return features