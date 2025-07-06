/**
 * Utility functions for parsing CSV data into 2D arrays of numbers
 */

export interface CSVParseResult {
  rows: number[][];
  errors: string[];
  warnings: string[];
}

export function validateCSVData(data: number[][]): boolean | string {
  // Check if data contains a NaN value or is empty
  if (!Array.isArray(data) || data.length === 0) {
    return "CSV is empty";
  }
  let expectedColumns: number | null = null;
  for (const rowIndex in data) {
    const row = data[rowIndex];
    expectedColumns = expectedColumns ?? row.length;
    if (!Array.isArray(row) || row.some((val) => isNaN(val))) {
      return "CSV contains NaN values";
    }
    if (row.length !== expectedColumns) {
      return `Row ${rowIndex} has ${row.length} columns, expected ${expectedColumns}`;
    }
  }
  return true;
}

/**
 * Parse CSV-like text into 2D array of numbers with detailed error reporting
 */
export function parseCSVData(text: string | number[][]): number[][] {
  if (Array.isArray(text)) return text;
  const rows: number[][] = [];

  const lines = text.trim().split("\n");

  for (let lineIndex = 0; lineIndex < lines.length; lineIndex++) {
    const line = lines[lineIndex].trim();

    // Skip empty lines
    if (!line) {
      continue;
    }

    // Parse values from line
    const rawValues = line.split(",").map((val) => val.trim());
    const values: number[] = [];

    for (let colIndex = 0; colIndex < rawValues.length; colIndex++) {
      const rawValue = rawValues[colIndex];
      const num = parseFloat(rawValue);
      if (isNaN(num) && lineIndex === 0) {
        values.length = 0;
        break;
      }

      values.push(num);
    }
    if (values.length === 0) continue;
    rows.push(values);
  }

  return rows;
}

/**
 * Format 2D array back to CSV-like text
 */
export function formatCSVData(data: number[][]): string {
  if (!data || data.length === 0) {
    return "";
  }

  return data.map((row) => row.join(",")).join("\n");
}
