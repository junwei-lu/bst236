"""
# 1. Two Sum (Easy)
# ----------------

# Problem Description:
# Given an array of integers 'nums' and an integer 'target', return indices of two numbers 
# that add up to 'target'. Each input has exactly one solution and you cannot use the 
# same element twice. The answer can be returned in any order.

# Examples:
# --------
# Example 1:
#   Input: nums = [2,7,11,15], target = 9
#   Output: [0,1]
#   Explanation: nums[0] + nums[1] = 2 + 7 = 9

# Example 2:
#   Input: nums = [3,2,4], target = 6 
#   Output: [1,2]

# Example 3:
#   Input: nums = [3,3], target = 6
#   Output: [0,1]

# Constraints:
# -----------
# - 2 <= nums.length <= 10^4
# - -10^9 <= nums[i] <= 10^9
# - -10^9 <= target <= 10^9
# - Only one valid answer exists

# Follow-up Question:
# Can you implement a solution with time complexity better than O(n^2)?
"""
import timeit
from typing import List

class Solution:
    def twoSum1(self, nums: List[int], target: int) -> List[int]:
        n = len(nums)
        for i in range(n):
            for j in range(n):
                if i != j:
                    if nums[i]+nums[j] == target:
                        return [i,j]

    def twoSum2(self, nums: List[int], target: int) -> List[int]:
        n = len(nums)
        for i in range(n):
            for j in range(i+1, n):
                if nums[i]+nums[j] == target:
                    return [i,j]     

    def twoSum3(self, nums: List[int], target: int) -> List[int]:
        n = len(nums)
        for i in range(n):
            v = target - nums[i]
            if v in nums[i+1:]:
                j = nums[i+1:].index(v)
                return [i,i+j+1]     
    
    def twoSum4(self, nums: List[int], target: int) -> List[int]:
        n = len(nums)
        index = {}
        for i in range(n):
            v = target - nums[i]
            if v in index:
                return [i,index[v]]
            index[nums[i]] = i  

    def twoSum5(self, nums: List[int], target: int) -> List[int]:
        n = len(nums)
        index = {}
        for i in range(n):
            v = target - nums[i]
            idx = index.get(v)
            if idx != None:
                return [i,idx]
            index[nums[i]] = i      

# generate example and test the time for each method
def twoSum_dict(nums, target):
        n = len(nums)
        index = {}
        for i in range(n):
            v = target - nums[i]
            if v in index:
                return [i,index[v]]
            index[nums[i]] = i 

def twoSum_list(nums, target):
        n = len(nums)
        for i in range(n):
            v = target - nums[i]
            if v in nums[i+1:]:
                j = nums[i+1:].index(v)
                return [i,i+j+1]  
 

nums = [2,7,11,15]
target = 26 
res = twoSum_list(nums, target)
res = twoSum_dict(nums, target)
