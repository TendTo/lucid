export default function Footer() {
  return (
    <footer className="bg-gray-800 text-white py-4 text-center">
      <p className="text-sm">
        &copy; {new Date().getFullYear()} Your Company Name. All rights
        reserved.
      </p>
      <p className="text-xs mt-2">Built with ❤️ using React and TypeScript.</p>
    </footer>
  );
}
