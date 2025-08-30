import React, { useState } from 'react';
import { z } from 'zod';
import { sendContact } from '../lib/api';
import Toast from '../components/Toast';

const schema = z.object({
  name: z.string().min(1),
  email: z.string().email(),
  message: z.string().min(5),
});

export default function Contact() {
  const [form, setForm] = useState({ name: '', email: '', message: '' });
  const [toast, setToast] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const res = schema.safeParse(form);
    if (!res.success) return alert('Please fill all fields correctly');
    await sendContact(form);
    setToast('Message sent!');
    setForm({ name: '', email: '', message: '' });
    setTimeout(() => setToast(''), 3000);
  };

  return (
    <div className="container mx-auto p-4 max-w-lg">
      <h1 className="text-2xl font-bold mb-4">Contact Us</h1>
      <form className="space-y-4" onSubmit={handleSubmit}>
        <input
          className="border p-2 w-full rounded"
          placeholder="Name"
          value={form.name}
          onChange={(e) => setForm({ ...form, name: e.target.value })}
        />
        <input
          className="border p-2 w-full rounded"
          placeholder="Email"
          value={form.email}
          onChange={(e) => setForm({ ...form, email: e.target.value })}
        />
        <textarea
          className="border p-2 w-full rounded"
          placeholder="Message"
          rows={4}
          value={form.message}
          onChange={(e) => setForm({ ...form, message: e.target.value })}
        />
        <button
          className="bg-blue-500 text-white px-4 py-2 rounded"
          type="submit"
        >
          Send
        </button>
      </form>
      <Toast message={toast} />
    </div>
  );
}
