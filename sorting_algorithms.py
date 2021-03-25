"""Implementation of popular sorting algorithms with automated testing."""
import time
import random
import pandas as pd
from statistics import median


class FileLoader():
    """Static class for loading integers or words from a .txt file."""

    @staticmethod
    def load_num(filename, lines):
        """Return a list of given number of integers from a file."""
        file = open(filename, 'r')
        return [int(line) for line in random.sample(file.readlines(), lines)]

    @staticmethod
    def load_str(filename, words):
        """Return a list of given number of words from a file."""
        file = open(filename, 'r')
        return [word[:-1] for word in random.sample(file.readlines(), words)]


class Sorter():
    """Static class wrapping all sorting methods."""

    @classmethod
    def _bin_search(cls, arr, start, end, key):
        """Return the index of the key in the array."""
        # Subarray is one element - return the index
        if start == end:
            return start
            # Calculate the index in between
        mid = start + int((end-start) / 2)
        # The searched value is either in the left or right subarray
        if key > arr[mid]:
            return cls._bin_search(arr, mid+1, end, key)
        if key < arr[mid]:
            return cls._bin_search(arr, start, mid, key)
        # If the searched value is exactly at mid index, return it
        return mid

    @staticmethod
    def is_sorted(arr):
        """Return whether the passed array is sorted or not."""
        for i in range(len(arr)-1):
            if arr[i] > arr[i+1]:
                return False
        return True

    @staticmethod
    def selection_sort(arr):
        """Return the array sorted using selection sort."""
        # For each successive element in the array, swap it
        # with the minimal element in the rest of the array.
        arr = arr.copy()
        for i in range(len(arr)):
            i_min = i
            for j in range(i+1, len(arr)):
                if arr[j] < arr[i_min]:
                    i_min = j
            arr[i], arr[i_min] = arr[i_min], arr[i]
        return arr

    @staticmethod
    def insertion_sort(arr):
        """Return the array sorted using insertion sort."""
        # Place each element from the second one onwards
        # in proper place by shifting it to the left until
        # it's the first element or the next element is not greater.
        arr = arr.copy()
        for i in range(1, len(arr)):
            ins = arr[i]
            j = i - 1
            while j > 0 and ins < arr[j]:
                arr[j+1] = arr[j]
                j -= 1
            arr[j] = ins
        return arr

    @classmethod
    def binary_insertion_sort(cls, arr):
        """Return the array sorted using selection sort with binary search."""
        # Insertion sort, but the place to insert each element
        # is found using binary search.
        arr = arr.copy()
        for i in range(1, len(arr)):
            ins = arr[i]
            # Index for the item to be placed is
            # found by binary search
            target = cls._bin_search(arr, 0, i, ins)
            j = i - 1
            while j >= target:
                arr[j+1] = arr[j]
                j -= 1
            arr[target] = ins
        return arr

    @staticmethod
    def bubble_sort(arr):
        """Return the array sorted using bubble sort."""
        # Take each element and keep moving it to the right,
        # until it reaches the end or greater element.
        # O(n^2) complexity.
        arr = arr.copy()
        top = len(arr) - 1
        for _ in range(len(arr)):
            for j in range(top):
                if arr[j] > arr[j+1]:
                    arr[j], arr[j+1] = arr[j+1], arr[j]
            top -= 1
        return arr

    @staticmethod
    def coctail_sort(arr):
        """Return the array sorted using coctail sort."""
        arr = arr.copy()
        bottom = 0
        top = len(arr) - 1
        for _ in range(int(len(arr)/2)):
            # We fix the largest element on the right
            for j in range(bottom, top):
                if arr[j] > arr[j+1]:
                    arr[j], arr[j+1] = arr[j+1], arr[j]
            top -= 1

            # We fix the smallest element on the left
            for j in range(top, bottom, -1):
                if arr[j] < arr[j-1]:
                    arr[j], arr[j-1] = arr[j-1], arr[j]
            bottom += 1
        return arr

    # Returns the list of step sizes for subarrays selection in shell sort.
    @staticmethod
    def _get_h_values(subarrays_type, arr):
        h_values = []
        # Set the separation values
        if subarrays_type == 'halves':
            h = int(len(arr)/2)
            while h >= 1:
                h_values.append(h)
                h = int(h / 2)
            h_values.reverse()
        elif subarrays_type == 'thirds':
            h = int(len(arr)/3)
            while h > 1:
                h_values.append(h)
                h = int(h / 3)
            h_values.append(1)
            h_values.reverse()
        elif subarrays_type == 'sedgewick':
            h_values = [4**k + 3*2**(k-1) + 1 for k in range(12)
                        if 4**k + 3*2**(k-1) + 1 < len(arr)]
            h_values[0] = 1
        return h_values

    @classmethod
    def shell_sort(cls, arr, subarrays_type='halves'):
        """
        Return the array sorted using shell sort.

        type:
        'halves', 'thirds', 'sedgewick'
        Specify the method of selecting subarray sizes (the size of the step
        when taking consecutive subarrays can be taken from multiple sources;
        subarrays with different spread yield different complexities).
        """
        arr = arr.copy()
        h_values = cls._get_h_values(subarrays_type, arr)
        h_index = len(h_values) - 1
        while h_index >= 0:
            h = h_values[h_index]
            # Like insertion sort, but on subarrays
            for i in range(h, len(arr)):
                ins = arr[i]
                j = i
                # Shift whole subarray until 'ins' is in proper position
                while j >= h and ins < arr[j-h]:
                    arr[j] = arr[j-h]
                    j -= h
                # Plug the considered element in proper position
                arr[j] = ins
            h_index -= 1
        return arr

    # Returns the index of the pivot element for quick sort.
    @staticmethod
    def _select_pivot(piv_selection, arr):
        if piv_selection == 'first':
            # Select the first element as the pivot
            pivot_index = 0
        elif piv_selection == 'median':
            # Rouge case check
            if len(arr) == 2:
                pivot_index = 0
            else:
                # Find the index of the median of three random elements
                three = random.sample(range(len(arr)), 3)
                for i in three:
                    if arr[i] == median(list(map(lambda x: arr[x], three))):
                        pivot_index = i
                        break
        elif piv_selection == 'random':
            # Select a random element as the pivot
            pivot_index = random.randint(0, len(arr))
        return pivot_index

    @classmethod
    def quick_sort(cls, arr, piv_selection='first'):
        """
        Return the array sorted using quick sort.

        piv_selection:
        'first', 'median', 'random'
        Specify the selection of the pivot element from the unsorted array.
        """
        arr = arr.copy()
        # One-item array is already sorted
        if len(arr) <= 1:
            return arr
        # Select appropriate pivot element, as specified in function argument.
        pivot_index = cls._select_pivot(piv_selection, arr)
        # Swap pivot with the first element
        arr[0], arr[pivot_index] = arr[pivot_index], arr[0]
        pivot = arr[0]
        s = 0
        for i in range(1, len(arr)):
            # Put all elements smaller than pivot next to it (at '++s' index)
            if arr[i] < pivot:
                s += 1
                arr[s], arr[i] = arr[i], arr[s]
        # Swap the last swapped element with the pivot.
        # Now all items smaller than the pivot are on the left,
        # and all items greater than the pivot are on the right.
        arr[s], arr[0] = arr[0], arr[s]
        # Repeat the process for the subarray on the left of the pivot
        # and the one on the right of the pivot.
        arr[:s] = cls.quick_sort(arr[:s])
        arr[s+1:] = cls.quick_sort(arr[s+1:])
        return arr

    @classmethod
    def quick_sort_insertion(cls, arr):
        """Return the array sorted using quick sort with insertion sort."""
        arr = arr.copy()
        # One-item array is already sorted
        if len(arr) <= 1:
            return arr
        # Small enough arrays can be dealt with using binary insertion sort
        if len(arr) <= 4:
            return cls.binary_insertion_sort(arr)
        # Select the first element as the pivot
        pivot = arr[0]
        s = 0
        for i in range(1, len(arr)):
            # Put all elements smaller than pivot next to it (at '++s' index)
            if arr[i] < pivot:
                s += 1
                arr[s], arr[i] = arr[i], arr[s]
        # Swap the latest smaller element with the pivot.
        # Now all items smaller than the pivot are on the left,
        # and all items greater than the pivot are on the right.
        arr[s], arr[0] = arr[0], arr[s]
        # Repeat the process for the subarray on the left of the pivot
        # and the one on the right of the pivot.
        arr[:s] = cls.quick_sort_insertion(arr[:s])
        arr[s+1:] = cls.quick_sort_insertion(arr[s+1:])
        return arr


class Tester():
    """Class for testing the performance of various sorting algorithms."""

    def __init__(self, data_type):
        """
        Create an instance of the class suited to a given data type.

        data_type:
        int, str, float
        """
        self.arr = []
        self.data_type = data_type
        self.arr_size = 0

    def prepare_array(self, arr_size):
        """Import the specified number of speficied objects."""
        self.arr_size = arr_size
        if self.data_type == int:
            self.arr = FileLoader.load_num('data.txt', arr_size)
        elif self.data_type == str:
            self.arr = FileLoader.load_str('words.txt', arr_size)
        elif self.data_type == float:
            self.arr = list(map(lambda x: x * random.random(),
                                FileLoader.load_num('data.txt', arr_size)))
        # Output a description of the loaded array.
        print(f'\nArray size: {arr_size}')
        print(f'Data type: {self.data_type}')
        print('Status: '
              + f'{"sorted" if Sorter.is_sorted(self.arr) else "unsorted"}')

    def perform_test(self, sorting_alg):
        """Sort the created array using specified sorting algorithm."""
        # Tidy up and output the name of the tested algorithm.
        alg_name = sorting_alg.capitalize().replace('sort', 'sort,').strip(',')
        print(f'\n=== {alg_name} ===')

        # Select the chosen algorithm and perform the sorting.
        start = time.perf_counter()
        if sorting_alg == 'selection sort':
            result = Sorter.selection_sort(self.arr)
        elif sorting_alg == 'insertion sort':
            result = Sorter.insertion_sort(self.arr)
        elif sorting_alg == 'binary insertion sort':
            result = Sorter.binary_insertion_sort(self.arr)
        elif sorting_alg == 'bubble sort':
            result = Sorter.bubble_sort(self.arr)
        elif sorting_alg == 'coctail sort':
            result = Sorter.coctail_sort(self.arr)
        elif sorting_alg == 'shell sort halves':
            result = Sorter.shell_sort(self.arr)
        elif sorting_alg == 'shell sort thirds':
            result = Sorter.shell_sort(self.arr, 'thirds')
        elif sorting_alg == 'shell sort sedgewick':
            result = Sorter.shell_sort(self.arr, 'sedgewick')
        elif sorting_alg == 'quick sort first':
            result = Sorter.quick_sort(self.arr)
        elif sorting_alg == 'quick sort median':
            result = Sorter.quick_sort(self.arr, 'median')
        elif sorting_alg == 'quick sort random':
            result = Sorter.quick_sort(self.arr, 'random')
        elif sorting_alg == 'quick sort insertion':
            result = Sorter.quick_sort_insertion(self.arr)
        stop = time.perf_counter()
        # Save and output the time of the algorithm and the state of the array.
        performance = round(stop-start, 5)
        print(f'Elapsed time: {performance} seconds')
        print('Status: '
              + f'{"sorted" if Sorter.is_sorted(result) else "unsorted"}')
        # Return the pair: the name of the algorithm and its performance time.
        return (alg_name, performance)

    def summarize_performance(self, group='all'):
        """Present the comparison of performances of each algorithm."""
        # Define and store all algorithm names to be recognized by the tester.
        alg_names = ['selection sort',
                     'insertion sort',
                     'binary insertion sort',
                     'bubble sort',
                     'coctail sort',
                     'shell sort halves',
                     'shell sort thirds',
                     'shell sort sedgewick',
                     'quick sort first',
                     'quick sort median',
                     'quick sort random',
                     'quick sort insertion']
        # Determine the subclass of the algorithms.
        if group == 'all':
            tested_algs = alg_names
            sizes = [1000, 2000, 4000, 8000, 16000]
        elif group == 'quick':
            tested_algs = alg_names[5:]
            sizes = [16000, 32000, 64000, 128000, 256000, 512000, 1024000]
        elif group == 'basic':
            sizes = [1000, 2000, 4000, 8000, 16000]
            tested_algs = alg_names[:5]
        # Create a dataframe to hold the performances on different array sizes.
        timetable = pd.DataFrame().rename_axis(
            index='Algorithm', columns='Data size')
        # Iterate over orders of magnitude of the array size.
        for test_arr_size in sizes:
            summary = pd.Series(name=test_arr_size)
            self.prepare_array(test_arr_size)
            for name in tested_algs:
                alg_name, perf_time = int_tester.perform_test(name)
                summary[alg_name] = perf_time
            timetable = pd.concat([timetable, summary], axis=1)
        return timetable


if __name__ == "_main_":
    # Create an integer sorting tester...
    int_tester = Tester(int)
    # ...specify the testing data size...
    int_tester.prepare_array(5000)
    # ...and perform tests
    int_tester.perform_test('insertion sort')
    int_tester.perform_test('binary insertion sort')
    # Save results for later...
    algorithm, performance_time = int_tester.perform_test('insertion sort')
    print(f'{algorithm}: {performance_time} seconds')
    # ...or let the code do all the work for you!
    print(int_tester.summarize_performance('quick'))
