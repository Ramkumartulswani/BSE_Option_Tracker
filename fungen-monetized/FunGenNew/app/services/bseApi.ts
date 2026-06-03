const API_BASE_URL = 'https://api.bseindia.com/BseIndiaAPI/api/DerivOptionChain_IV/w';

interface RawOptionRow {
  Strike_Price1: string;
  Open_Interest: string;
  C_Open_Interest: string;
  Vol_Traded: string;
  C_Vol_Traded: string;
  Last_Trd_Price: string;
  C_Last_Trd_Price: string;
  IV: string;
  C_IV: string;
}

export interface OptionChainRow {
  'Strike Price': number;
  'CE OI': number;
  'CE LTP': number;
  'CE Volume': number;
  'CE IV': number;
  'PE OI': number;
  'PE LTP': number;
  'PE Volume': number;
  'PE IV': number;
}

export interface BseResponse {
  spot_price: number;
  day_high: number | null;
  day_low: number | null;
  rows: OptionChainRow[];
  expiry: string;
  scrip_cd: string;
}

interface BseApiResponse {
  Table: RawOptionRow[];
  UlaValue?: string;
  UnderlyingValue?: string;
  underlyingValue?: string;
  spotPrice?: string;
  High?: string;
  Low?: string;
  DayHigh?: string;
  DayLow?: string;
}

function safeFloat(value: any, defaultVal = 0): number {
  try {
    const v = parseFloat(String(value).replace(/,/g, '').trim());
    return isNaN(v) ? defaultVal : v;
  } catch {
    return defaultVal;
  }
}

function extractSpot(data: BseApiResponse, rows: OptionChainRow[]): number {
  const keys = ['UlaValue', 'UnderlyingValue', 'underlyingValue', 'Underlying_Value', 'spotPrice', 'SpotPrice', 'IndexValue', 'indexValue'];
  for (const k of keys) {
    const v = safeFloat((data as any)[k]);
    if (v > 0) return v;
  }
  const prices = rows.map(r => r['Strike Price']).filter(p => p > 0);
  if (prices.length > 0) {
    prices.sort((a, b) => a - b);
    return prices[Math.floor(prices.length / 2)];
  }
  return 0;
}

function extractVal(data: BseApiResponse, keys: string[]): number | null {
  for (const k of keys) {
    const v = safeFloat((data as any)[k]);
    if (v > 0) return v;
  }
  return null;
}

function processTable(table: RawOptionRow[]): OptionChainRow[] {
  const df = table.map(row => ({
    'Strike Price': safeFloat(row.Strike_Price1),
    'CE OI': safeFloat(row.C_Open_Interest),
    'CE LTP': safeFloat(row.C_Last_Trd_Price),
    'CE Volume': safeFloat(row.C_Vol_Traded),
    'CE IV': safeFloat(row.C_IV),
    'PE OI': safeFloat(row.Open_Interest),
    'PE LTP': safeFloat(row.Last_Trd_Price),
    'PE Volume': safeFloat(row.Vol_Traded),
    'PE IV': safeFloat(row.IV),
  }));
  return df
    .filter(r => r['CE OI'] > 0 || r['PE OI'] > 0)
    .sort((a, b) => a['Strike Price'] - b['Strike Price']);
}

const BSE_HEADERS = {
  accept: 'application/json, text/plain, */*',
  origin: 'https://www.bseindia.com',
  referer: 'https://www.bseindia.com/',
  'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
};

export async function fetchBseOptionChain(
  expiry: string,
  scripCd = '1',
  strprice = '0',
): Promise<BseResponse> {
  const url = `${API_BASE_URL}?Expiry=${encodeURIComponent(expiry)}&scrip_cd=${scripCd}&strprice=${strprice}`;
  const resp = await fetch(url, { headers: BSE_HEADERS });
  if (!resp.ok) {
    throw new Error(`HTTP ${resp.status}`);
  }
  const data: BseApiResponse = await resp.json();
  const table = data.Table || [];
  if (table.length === 0) {
    throw new Error('No data for given expiry.');
  }
  const rows = processTable(table);
  const spotPrice = extractSpot(data, rows);
  const dayHigh = extractVal(data, ['High', 'high', 'DayHigh', 'dayHigh']);
  const dayLow = extractVal(data, ['Low', 'low', 'DayLow', 'dayLow']);
  return { spot_price: spotPrice, day_high: dayHigh, day_low: dayLow, rows, expiry, scrip_cd: scripCd };
}

function generateDefaultExpiries(): string[] {
  const dates: string[] = [];
  const now = new Date();
  for (let i = 0; i < 12; i++) {
    let m = now.getMonth() + i;
    let y = now.getFullYear() + Math.floor(m / 12);
    m = ((m % 12) + 12) % 12;
    const lastDay = new Date(y, m + 1, 0);
    while (lastDay.getDay() !== 4) {
      lastDay.setDate(lastDay.getDate() - 1);
    }
    const dd = String(lastDay.getDate()).padStart(2, '0');
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    dates.push(`${dd} ${months[lastDay.getMonth()]} ${lastDay.getFullYear()}`);
  }
  return dates;
}

export async function fetchExpiryDates(scripCd = '1'): Promise<string[]> {
  const urls = [
    `https://api.bseindia.com/BseIndiaAPI/api/DDlExpiry/w?flag=0&scripcode=${scripCd}`,
    `https://api.bseindia.com/BseIndiaAPI/api/DefaultData/w?scripcode=${scripCd}`,
  ];
  for (const url of urls) {
    try {
      const resp = await fetch(url, { headers: BSE_HEADERS });
      if (resp.ok) {
        const data = await resp.json();
        const dates = extractDates(data);
        if (dates.length > 0) {
          dates.sort((a, b) => new Date(a).getTime() - new Date(b).getTime());
          return Array.from(new Set(dates));
        }
      }
    } catch {
      continue;
    }
  }
  return generateDefaultExpiries();
}

function extractDates(data: any): string[] {
  const dates: string[] = [];
  const keys = ['Table', 'expiry', 'expiryDate', 'ExpiryDates', 'Expiry', 'expDates', 'ExpiryList', 'expirylist', 'expiryDt'];
  for (const key of keys) {
    const arr = data[key];
    if (Array.isArray(arr)) {
      for (const item of arr) {
        if (typeof item === 'string' && item.trim()) {
          dates.push(item.trim());
        } else if (typeof item === 'object') {
          for (const dk of ['expiry', 'Expiry', 'ExpiryDate', 'expiry_date', 'Expiry_Date', 'expiryDt', 'ExpiryDt']) {
            if (item[dk] && String(item[dk]).trim()) {
              dates.push(String(item[dk]).trim());
            }
          }
        }
      }
    }
  }
  return dates;
}
