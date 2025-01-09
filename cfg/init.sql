-- Create database
CREATE DATABASE grafana_data;

-- Use the database
USE grafana_data;

-- Create table
CREATE TABLE metrics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    time_column DATETIME NOT NULL,
    Blackbird INT NOT NULL DEFAULT 0,
    Butcherbird INT NOT NULL DEFAULT 0,
    Currawong INT NOT NULL DEFAULT 0,
    Dove INT NOT NULL DEFAULT 0,
    Lorikeet INT NOT NULL DEFAULT 0,
    Myna INT NOT NULL DEFAULT 0,
    Sparrow INT NOT NULL DEFAULT 0,
    Starling INT NOT NULL DEFAULT 0,
    Wattlebird INT NOT NULL DEFAULT 0
);

-- Create user and grant permissions
CREATE USER 'grafana_user'@'%' IDENTIFIED BY 'grafana_password';
GRANT SELECT ON grafana_data.* TO 'grafana_user'@'%';
FLUSH PRIVILEGES;
