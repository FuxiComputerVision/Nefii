import torch


class SamplerGivenSeq(torch.utils.data.Sampler):
    def __init__(self, sample_seq):
        self.sample_seq = sample_seq

    def __iter__(self):
        return iter(self.sample_seq)

    def __len__(self):
        return len(self.sample_seq)


class SamplerRandomChoice(torch.utils.data.Sampler):
    def __init__(self, dataset, num, generator=None):
        self.num = num
        self.dataset = dataset
        self.generator = generator

    def __iter__(self):
        random_seq = torch.randperm(len(self.dataset), generator=self.generator)[:self.num]
        return iter(random_seq)

    def __len__(self):
        return self.num


class SamplerFixIndex(torch.utils.data.Sampler):
    def __init__(self, num, index=0):
        self.num = num
        self.index = index

    def __iter__(self):
        return FixIter(self.num, self.index)

    def __len__(self):
        return self.num


class FixIter:
    def __init__(self, num, index=0):
        self.num = num
        self.index = index
        self.count = 0

    def __next__(self):
        if self.count < self.num:
            return self.index

        else:
            raise StopIteration
