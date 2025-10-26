-- ====================================
-- PhonePe Pulse Database Schema
-- ====================================

-- Create Database
CREATE DATABASE IF NOT EXISTS phonepe_pulse;
USE phonepe_pulse;

-- ====================================
-- AGGREGATED TABLES
-- ====================================

-- Aggregated Transaction Table
CREATE TABLE IF NOT EXISTS aggregated_transaction (
    id INT AUTO_INCREMENT PRIMARY KEY,
    state VARCHAR(100) NOT NULL,
    year INT NOT NULL,
    quarter INT NOT NULL,
    transaction_type VARCHAR(100) NOT NULL,
    transaction_count BIGINT NOT NULL,
    transaction_amount DECIMAL(20, 2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_state_year_quarter (state, year, quarter),
    INDEX idx_year_quarter (year, quarter),
    INDEX idx_transaction_type (transaction_type)
);

-- Aggregated User Table
CREATE TABLE IF NOT EXISTS aggregated_user (
    id INT AUTO_INCREMENT PRIMARY KEY,
    state VARCHAR(100) NOT NULL,
    year INT NOT NULL,
    quarter INT NOT NULL,
    registered_users BIGINT NOT NULL,
    app_opens BIGINT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_state_year_quarter (state, year, quarter),
    INDEX idx_year_quarter (year, quarter)
);

-- Aggregated Insurance Table
CREATE TABLE IF NOT EXISTS aggregated_insurance (
    id INT AUTO_INCREMENT PRIMARY KEY,
    state VARCHAR(100) NOT NULL,
    year INT NOT NULL,
    quarter INT NOT NULL,
    transaction_count BIGINT NOT NULL,
    transaction_amount DECIMAL(20, 2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_state_year_quarter (state, year, quarter),
    INDEX idx_year_quarter (year, quarter)
);

-- ====================================
-- MAP TABLES (District Level Data)
-- ====================================

-- Map Transaction Table
CREATE TABLE IF NOT EXISTS map_transaction (
    id INT AUTO_INCREMENT PRIMARY KEY,
    state VARCHAR(100) NOT NULL,
    district VARCHAR(100) NOT NULL,
    year INT NOT NULL,
    quarter INT NOT NULL,
    transaction_count BIGINT NOT NULL,
    transaction_amount DECIMAL(20, 2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_state_district (state, district),
    INDEX idx_year_quarter (year, quarter)
);

-- Map User Table
CREATE TABLE IF NOT EXISTS map_user (
    id INT AUTO_INCREMENT PRIMARY KEY,
    state VARCHAR(100) NOT NULL,
    district VARCHAR(100) NOT NULL,
    year INT NOT NULL,
    quarter INT NOT NULL,
    registered_users BIGINT NOT NULL,
    app_opens BIGINT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_state_district (state, district),
    INDEX idx_year_quarter (year, quarter)
);

-- Map Insurance Table
CREATE TABLE IF NOT EXISTS map_insurance (
    id INT AUTO_INCREMENT PRIMARY KEY,
    state VARCHAR(100) NOT NULL,
    district VARCHAR(100) NOT NULL,
    year INT NOT NULL,
    quarter INT NOT NULL,
    transaction_count BIGINT NOT NULL,
    transaction_amount DECIMAL(20, 2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_state_district (state, district),
    INDEX idx_year_quarter (year, quarter)
);

-- ====================================
-- TOP TABLES (Ranking Data)
-- ====================================

-- Top Transaction Table (States, Districts, Pincodes)
CREATE TABLE IF NOT EXISTS top_transaction (
    id INT AUTO_INCREMENT PRIMARY KEY,
    category VARCHAR(20) NOT NULL, -- 'states', 'districts', 'pincodes'
    entity_name VARCHAR(100) NOT NULL,
    year INT NOT NULL,
    quarter INT NOT NULL,
    transaction_count BIGINT NOT NULL,
    transaction_amount DECIMAL(20, 2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_category_year_quarter (category, year, quarter),
    INDEX idx_year_quarter (year, quarter)
);

-- Top User Table
CREATE TABLE IF NOT EXISTS top_user (
    id INT AUTO_INCREMENT PRIMARY KEY,
    category VARCHAR(20) NOT NULL,
    entity_name VARCHAR(100) NOT NULL,
    year INT NOT NULL,
    quarter INT NOT NULL,
    registered_users BIGINT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_category_year_quarter (category, year, quarter)
);

-- Top Insurance Table
CREATE TABLE IF NOT EXISTS top_insurance (
    id INT AUTO_INCREMENT PRIMARY KEY,
    category VARCHAR(20) NOT NULL,
    entity_name VARCHAR(100) NOT NULL,
    year INT NOT NULL,
    quarter INT NOT NULL,
    transaction_count BIGINT NOT NULL,
    transaction_amount DECIMAL(20, 2) NOT NULL,
    created_at TIMESTAMP DEFAULT