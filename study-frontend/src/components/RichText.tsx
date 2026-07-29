import ReactMarkdown, { type Components } from "react-markdown"

const components: Components = {
  h1: ({ children }) => <h1 className="mb-3 mt-7 text-2xl font-semibold first:mt-0">{children}</h1>,
  h2: ({ children }) => <h2 className="mb-3 mt-7 text-xl font-semibold first:mt-0">{children}</h2>,
  h3: ({ children }) => <h3 className="mb-2 mt-6 text-lg font-semibold first:mt-0">{children}</h3>,
  p: ({ children }) => <p className="mb-4 last:mb-0">{children}</p>,
  ul: ({ children }) => <ul className="mb-4 list-disc space-y-2 pl-6 last:mb-0">{children}</ul>,
  ol: ({ children }) => <ol className="mb-4 list-decimal space-y-2 pl-6 last:mb-0">{children}</ol>,
  li: ({ children }) => <li className="pl-1">{children}</li>,
  strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
  a: ({ children, href }) => {
    const external = href?.startsWith("http")
    return (
      <a className="font-medium text-primary underline underline-offset-2" href={href}
        target={external ? "_blank" : undefined} rel={external ? "noreferrer" : undefined}>
        {children}
      </a>
    )
  },
}

export function RichText({ children, className = "" }: { children: string; className?: string }) {
  return (
    <div className={`text-base leading-7 text-foreground/90 ${className}`}>
      <ReactMarkdown components={components}>{children}</ReactMarkdown>
    </div>
  )
}
