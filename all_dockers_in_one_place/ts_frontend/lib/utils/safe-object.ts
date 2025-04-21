/**
 * Safely get entries from an object, returning an empty array if the object is null or undefined
 * @param obj The object to get entries from
 * @returns Array of [key, value] pairs, or empty array if obj is null/undefined
 */
export function safeObjectEntries<T = any>(obj: Record<string, T> | null | undefined): [string, T][] {
  if (obj === null || obj === undefined) {
    return [];
  }
  return Object.entries(obj);
}

/**
 * Safely get keys from an object, returning an empty array if the object is null or undefined
 * @param obj The object to get keys from
 * @returns Array of keys, or empty array if obj is null/undefined
 */
export function safeObjectKeys(obj: Record<string, any> | null | undefined): string[] {
  if (obj === null || obj === undefined) {
    return [];
  }
  return Object.keys(obj);
}

/**
 * Safely get values from an object, returning an empty array if the object is null or undefined
 * @param obj The object to get values from
 * @returns Array of values, or empty array if obj is null/undefined
 */
export function safeObjectValues<T = any>(obj: Record<string, T> | null | undefined): T[] {
  if (obj === null || obj === undefined) {
    return [];
  }
  return Object.values(obj);
}