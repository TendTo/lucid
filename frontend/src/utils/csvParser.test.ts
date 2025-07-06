/**
 * Test file for CSV parser utility
 */
import { parseCSVData, formatCSVData } from './csvParser';

// Test basic parsing
console.log('Testing basic CSV parsing:');
const testData1 = `1.0,2.0,3.0
4.0,5.0,6.0
7.0,8.0,9.0`;

const result1 = parseCSVData(testData1);
console.log('Result:', result1);
console.log('Expected: [[1,2,3], [4,5,6], [7,8,9]]');

// Test with errors
console.log('\nTesting CSV with errors:');
const testData2 = `1.0,2.0,3.0
4.0,invalid,6.0
7.0,8.0`;

const result2 = parseCSVData(testData2);
console.log('Result:', result2);

// Test formatting
console.log('\nTesting formatting:');
const testArray = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
const formatted = formatCSVData(testArray);
console.log('Formatted:', formatted);
console.log('Expected: "1,2,3\\n4,5,6\\n7,8,9"');

