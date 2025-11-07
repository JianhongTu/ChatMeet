//
//  APIKeyManager.swift
//  MeetingAssistantCore
//
//  Secure storage for API keys using Keychain
//

import Foundation
import Security

/// Manager for securely storing and retrieving API keys in Keychain
public class APIKeyManager {
    
    public static let shared = APIKeyManager()
    
    private let service = "com.chatmeet.api"
    private let apiKeyAccount = "openai_api_key"
    private let endpointAccount = "api_endpoint"
    private let modelAccount = "model_name"
    
    private init() {}
    
    // MARK: - API Key Management
    
    /// Save API key to Keychain
    /// - Parameter apiKey: The API key to store
    /// - Returns: True if successful, false otherwise
    @discardableResult
    public func saveAPIKey(_ apiKey: String) -> Bool {
        return saveToKeychain(account: apiKeyAccount, value: apiKey)
    }
    
    /// Retrieve API key from Keychain
    /// - Returns: The stored API key, or nil if not found
    public func getAPIKey() -> String? {
        return retrieveFromKeychain(account: apiKeyAccount)
    }
    
    /// Delete API key from Keychain
    /// - Returns: True if successful, false otherwise
    @discardableResult
    public func deleteAPIKey() -> Bool {
        return deleteFromKeychain(account: apiKeyAccount)
    }
    
    // MARK: - Endpoint Management
    
    /// Save API endpoint to Keychain
    /// - Parameter endpoint: The API endpoint URL
    /// - Returns: True if successful, false otherwise
    @discardableResult
    public func saveAPIEndpoint(_ endpoint: String) -> Bool {
        return saveToKeychain(account: endpointAccount, value: endpoint)
    }
    
    /// Retrieve API endpoint from Keychain
    /// - Returns: The stored endpoint, or nil if not found
    public func getAPIEndpoint() -> String? {
        return retrieveFromKeychain(account: endpointAccount)
    }
    
    // MARK: - Model Name Management
    
    /// Save model name to Keychain
    /// - Parameter modelName: The model name to use
    /// - Returns: True if successful, false otherwise
    @discardableResult
    public func saveModelName(_ modelName: String) -> Bool {
        return saveToKeychain(account: modelAccount, value: modelName)
    }
    
    /// Retrieve model name from Keychain
    /// - Returns: The stored model name, or nil if not found
    public func getModelName() -> String? {
        return retrieveFromKeychain(account: modelAccount)
    }
    
    // MARK: - Keychain Operations
    
    private func saveToKeychain(account: String, value: String) -> Bool {
        guard let data = value.data(using: .utf8) else {
            return false
        }
        
        // Delete any existing item first
        deleteFromKeychain(account: account)
        
        // Create new keychain item
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlocked
        ]
        
        let status = SecItemAdd(query as CFDictionary, nil)
        
        if status == errSecSuccess {
            print("✅ APIKeyManager: Saved \(account)")
            return true
        } else {
            print("❌ APIKeyManager: Failed to save \(account), status: \(status)")
            return false
        }
    }
    
    private func retrieveFromKeychain(account: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]
        
        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        
        guard status == errSecSuccess,
              let data = result as? Data,
              let value = String(data: data, encoding: .utf8) else {
            if status != errSecItemNotFound {
                print("⚠️ APIKeyManager: Failed to retrieve \(account), status: \(status)")
            }
            return nil
        }
        
        return value
    }
    
    private func deleteFromKeychain(account: String) -> Bool {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account
        ]
        
        let status = SecItemDelete(query as CFDictionary)
        
        // Both success and item-not-found are considered "successful" deletion
        return status == errSecSuccess || status == errSecItemNotFound
    }
    
    /// Check if API key is configured
    public func hasAPIKey() -> Bool {
        return getAPIKey() != nil
    }
    
    /// Clear all stored credentials
    public func clearAll() {
        deleteAPIKey()
        deleteFromKeychain(account: endpointAccount)
        deleteFromKeychain(account: modelAccount)
        print("🗑️ APIKeyManager: Cleared all credentials")
    }
}
