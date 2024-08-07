def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

# 예제 리스트
arr = [64, 25, 12, 22, 11]
print(f"Original list: {arr}")
sorted_arr = bubble_sort(arr)
print(f"Sorted list: {sorted_arr}")