//
//  TokenMerger.swift
//  TranscriptionServices
//
//  FluidAudio-style 3-tier token merging for chunk boundaries
//  Strategy: Contiguous pairs → LCS → Midpoint split
//

import Foundation

/// Merges tokens from overlapping chunks using FluidAudio's 3-tier strategy
class TokenMerger {
    
    private let sampleRate: Int = 16000
    private let samplesPerFrame: Int = 1280  // Parakeet encoder frame size (80ms)
    
    /// Merge two token arrays from overlapping chunks
    /// - Parameters:
    ///   - left: Tokens from first chunk
    ///   - right: Tokens from second chunk
    ///   - overlapSeconds: Overlap duration in seconds
    /// - Returns: Merged token array
    func mergeChunks(
        _ left: [TokenWindow],
        _ right: [TokenWindow],
        overlapSeconds: Double
    ) -> [TokenWindow] {
        guard !left.isEmpty else { return right }
        guard !right.isEmpty else { return left }
        
        let frameDuration = Double(samplesPerFrame) / Double(sampleRate)
        let halfOverlapWindow = overlapSeconds / 2.0
        
        // Calculate temporal boundaries
        let leftEndFrame = left.last!.timestamp
        let rightStartFrame = right.first!.timestamp
        let leftEndTime = Double(leftEndFrame) * frameDuration
        let rightStartTime = Double(rightStartFrame) * frameDuration
        
        // No temporal overlap - just concatenate
        if leftEndTime <= rightStartTime {
            return left + right
        }
        
        // Find tokens in overlap region
        let overlapLeft = findOverlapTokens(
            in: left,
            thresholdFrame: leftEndFrame,
            overlapFrames: Int(overlapSeconds / frameDuration),
            searchFromEnd: true
        )
        
        let overlapRight = findOverlapTokens(
            in: right,
            thresholdFrame: rightStartFrame,
            overlapFrames: Int(overlapSeconds / frameDuration),
            searchFromEnd: false
        )
        
        // Need at least 2 tokens on each side for matching
        guard overlapLeft.count >= 2 && overlapRight.count >= 2 else {
            return mergeByMidpoint(
                left: left,
                right: right,
                leftEndFrame: leftEndFrame,
                rightStartFrame: rightStartFrame,
                frameDuration: frameDuration
            )
        }
        
        // Strategy 1: Find best contiguous pairs (ideal case)
        let minimumPairs = max(overlapLeft.count / 2, 1)
        let contiguousPairs = findBestContiguousPairs(
            overlapLeft: overlapLeft,
            overlapRight: overlapRight,
            tolerance: halfOverlapWindow,
            frameDuration: frameDuration
        )
        
        if contiguousPairs.count >= minimumPairs {
            return mergeUsingMatches(
                matches: contiguousPairs,
                overlapLeft: overlapLeft,
                overlapRight: overlapRight,
                left: left,
                right: right
            )
        }
        
        // Strategy 2: LCS fallback when contiguous insufficient
        let lcsPairs = findLongestCommonSubsequence(
            overlapLeft: overlapLeft,
            overlapRight: overlapRight,
            tolerance: halfOverlapWindow,
            frameDuration: frameDuration
        )
        
        guard !lcsPairs.isEmpty else {
            return mergeByMidpoint(
                left: left,
                right: right,
                leftEndFrame: leftEndFrame,
                rightStartFrame: rightStartFrame,
                frameDuration: frameDuration
            )
        }
        
        return mergeUsingMatches(
            matches: lcsPairs,
            overlapLeft: overlapLeft,
            overlapRight: overlapRight,
            left: left,
            right: right
        )
    }
    
    // MARK: - Helper Structures
    
    private struct IndexedToken {
        let index: Int
        let token: TokenWindow
        let timeSeconds: Double
    }
    
    private func findOverlapTokens(
        in tokens: [TokenWindow],
        thresholdFrame: Int,
        overlapFrames: Int,
        searchFromEnd: Bool
    ) -> [IndexedToken] {
        let frameDuration = Double(samplesPerFrame) / Double(sampleRate)
        var result: [IndexedToken] = []
        
        for (index, token) in tokens.enumerated() {
            let timeSeconds = Double(token.timestamp) * frameDuration
            let inOverlap: Bool
            
            if searchFromEnd {
                // Looking for tokens near end of left chunk
                inOverlap = token.timestamp >= (thresholdFrame - overlapFrames)
            } else {
                // Looking for tokens near start of right chunk
                inOverlap = token.timestamp <= (thresholdFrame + overlapFrames)
            }
            
            if inOverlap {
                result.append(IndexedToken(index: index, token: token, timeSeconds: timeSeconds))
            }
        }
        
        return result
    }
    
    private func tokensMatch(
        _ left: IndexedToken,
        _ right: IndexedToken,
        tolerance: Double
    ) -> Bool {
        guard left.token.token == right.token.token else { return false }
        let timeDifference = abs(left.timeSeconds - right.timeSeconds)
        return timeDifference < tolerance
    }
    
    // MARK: - Strategy 1: Contiguous Pairs
    
    private func findBestContiguousPairs(
        overlapLeft: [IndexedToken],
        overlapRight: [IndexedToken],
        tolerance: Double,
        frameDuration: Double
    ) -> [(Int, Int)] {
        var best: [(Int, Int)] = []
        
        for i in 0..<overlapLeft.count {
            for j in 0..<overlapRight.count {
                if tokensMatch(overlapLeft[i], overlapRight[j], tolerance: tolerance) {
                    var current: [(Int, Int)] = []
                    var k = i
                    var l = j
                    
                    // Extend contiguous match as far as possible
                    while k < overlapLeft.count && l < overlapRight.count {
                        if tokensMatch(overlapLeft[k], overlapRight[l], tolerance: tolerance) {
                            current.append((k, l))
                            k += 1
                            l += 1
                        } else {
                            break
                        }
                    }
                    
                    if current.count > best.count {
                        best = current
                    }
                }
            }
        }
        
        return best
    }
    
    // MARK: - Strategy 2: Longest Common Subsequence
    
    private func findLongestCommonSubsequence(
        overlapLeft: [IndexedToken],
        overlapRight: [IndexedToken],
        tolerance: Double,
        frameDuration: Double
    ) -> [(Int, Int)] {
        let leftCount = overlapLeft.count
        let rightCount = overlapRight.count
        
        // DP table
        var dp = Array(repeating: Array(repeating: 0, count: rightCount + 1), count: leftCount + 1)
        
        // Fill DP table
        for i in 1...leftCount {
            for j in 1...rightCount {
                if tokensMatch(overlapLeft[i - 1], overlapRight[j - 1], tolerance: tolerance) {
                    dp[i][j] = dp[i - 1][j - 1] + 1
                } else {
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                }
            }
        }
        
        // Backtrack to find pairs
        var pairs: [(Int, Int)] = []
        var i = leftCount
        var j = rightCount
        
        while i > 0 && j > 0 {
            if tokensMatch(overlapLeft[i - 1], overlapRight[j - 1], tolerance: tolerance) {
                pairs.append((i - 1, j - 1))
                i -= 1
                j -= 1
            } else if dp[i - 1][j] > dp[i][j - 1] {
                i -= 1
            } else {
                j -= 1
            }
        }
        
        return pairs.reversed()
    }
    
    // MARK: - Strategy 3: Midpoint Split
    
    private func mergeByMidpoint(
        left: [TokenWindow],
        right: [TokenWindow],
        leftEndFrame: Int,
        rightStartFrame: Int,
        frameDuration: Double
    ) -> [TokenWindow] {
        let midpointFrame = (leftEndFrame + rightStartFrame) / 2
        
        // Take all left tokens before midpoint
        let leftPart = left.filter { $0.timestamp < midpointFrame }
        
        // Take all right tokens at or after midpoint
        let rightPart = right.filter { $0.timestamp >= midpointFrame }
        
        return leftPart + rightPart
    }
    
    // MARK: - Merge Using Matches
    
    private func mergeUsingMatches(
        matches: [(Int, Int)],
        overlapLeft: [IndexedToken],
        overlapRight: [IndexedToken],
        left: [TokenWindow],
        right: [TokenWindow]
    ) -> [TokenWindow] {
        guard !matches.isEmpty else {
            return left + right
        }
        
        print("🔍 MERGE DEBUG:")
        print("  Left: \(left.count) tokens (frames \(left.first?.timestamp ?? -1)...\(left.last?.timestamp ?? -1))")
        print("  Right: \(right.count) tokens (frames \(right.first?.timestamp ?? -1)...\(right.last?.timestamp ?? -1))")
        print("  Overlap left: \(overlapLeft.count) tokens")
        if !overlapLeft.isEmpty {
            print("    First overlap left: index=\(overlapLeft.first!.index), frame=\(overlapLeft.first!.token.timestamp)")
            print("    Last overlap left: index=\(overlapLeft.last!.index), frame=\(overlapLeft.last!.token.timestamp)")
        }
        print("  Overlap right: \(overlapRight.count) tokens")
        if !overlapRight.isEmpty {
            print("    First overlap right: index=\(overlapRight.first!.index), frame=\(overlapRight.first!.token.timestamp)")
            print("    Last overlap right: index=\(overlapRight.last!.index), frame=\(overlapRight.last!.token.timestamp)")
        }
        print("  Matches: \(matches.count) pairs")
        
        // New simpler strategy using temporal boundaries:
        // 1. Determine the overlap region's temporal boundaries
        // 2. Keep ALL of left that's BEFORE the overlap region
        // 3. Keep ALL of right (overlap region has better context, so right's version is more accurate)
        
        // Find the temporal boundary - where does the overlap region start?
        // Use the first token in right's overlap as the cutoff point
        guard let firstOverlapRight = overlapRight.first else {
            // No overlap tokens in right, just concatenate
            return left + right
        }
        
        let overlapStartFrame = firstOverlapRight.token.timestamp
        
        print("  Overlap region starts at frame \(overlapStartFrame)")
        
        // Keep all left tokens that come BEFORE the overlap region
        // These are definitely not duplicates
        let leftPart = left.filter { $0.timestamp < overlapStartFrame }
        
        // Keep ALL right tokens
        // Right has better context (15s vs whatever was in left's overlap)
        let rightPart = right
        
        print("  Result: left[\(leftPart.count)] + right[\(rightPart.count)] = \(leftPart.count + rightPart.count) tokens")
        print("  Discarded \(left.count - leftPart.count) tokens from left's overlap region")
        
        return leftPart + rightPart
    }
}