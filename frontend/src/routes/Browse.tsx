import React, { useEffect, useState } from 'react';
import { DestinationList } from '../lib/types';
import { getDestinations } from '../lib/api';
import Card from '../components/Card';
import Skeleton from '../components/Skeleton';

export default function Browse() {
  const [data, setData] = useState<DestinationList | null>(null);
  const [loading, setLoading] = useState(false);
  const [q, setQ] = useState('');
  const [country, setCountry] = useState('');
  const [sort, setSort] = useState('rating');
  const [page, setPage] = useState(1);
  const [countries, setCountries] = useState<string[]>([]);

  useEffect(() => {
    getDestinations({ limit: 100 }).then((res) => {
      const c = Array.from(new Set(res.items.map((i) => i.country))).sort();
      setCountries(c);
    });
  }, []);

  useEffect(() => {
    setLoading(true);
    const t = setTimeout(() => {
      getDestinations({ q, country, sort, page }).then((res) => {
        setData(res);
        setLoading(false);
      });
    }, 300);
    return () => clearTimeout(t);
  }, [q, country, sort, page]);

  return (
    <div className="container mx-auto p-4 space-y-4">
      <div className="flex flex-col md:flex-row gap-2 md:items-center">
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search..."
          className="border p-2 rounded w-full md:w-1/3"
        />
        <select
          value={country}
          onChange={(e) => {
            setCountry(e.target.value);
            setPage(1);
          }}
          className="border p-2 rounded"
        >
          <option value="">All countries</option>
          {countries.map((c) => (
            <option key={c}>{c}</option>
          ))}
        </select>
        <select
          value={sort}
          onChange={(e) => setSort(e.target.value)}
          className="border p-2 rounded"
        >
          <option value="name">Name</option>
          <option value="rating">Rating</option>
        </select>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
        {loading &&
          Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} />)}
        {!loading && data?.items.map((d) => <Card key={d.id} dest={d} />)}
      </div>
      <div className="flex justify-center gap-4">
        <button
          onClick={() => setPage((p) => Math.max(1, p - 1))}
          disabled={page === 1}
          className="px-3 py-1 bg-blue-500 text-white rounded disabled:opacity-50"
        >
          Prev
        </button>
        <span>
          Page {data?.page || page} / {data?.totalPages || 1}
        </span>
        <button
          onClick={() =>
            setPage((p) => (data ? Math.min(data.totalPages, p + 1) : p + 1))
          }
          disabled={data ? page >= data.totalPages : false}
          className="px-3 py-1 bg-blue-500 text-white rounded disabled:opacity-50"
        >
          Next
        </button>
      </div>
    </div>
  );
}
