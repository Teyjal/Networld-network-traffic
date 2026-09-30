import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Outlet, Link, createRootRouteWithContext, useRouter, HeadContent, Scripts } from "@tanstack/react-router";
import { useEffect, type ReactNode } from "react";
import appCss from "../styles.css?url";
import { reportLovableError } from "../lib/lovable-error-reporting";
import { AppShell } from "@/components/layout/AppShell";
function NotFoundComponent(){return <div className="flex min-h-screen items-center justify-center bg-background px-4"><div className="text-center"><p className="eyebrow">SYSTEM RESPONSE</p><h1 className="mt-3 font-display text-7xl text-primary">404</h1><p className="mt-2 text-muted-foreground">The requested intelligence view is unavailable.</p><Link to="/" className="mt-6 inline-flex rounded-md bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground">Return to command center</Link></div></div>}
function ErrorComponent({error,reset}:{error:Error;reset:()=>void}){const router=useRouter();useEffect(()=>{reportLovableError(error,{boundary:"tanstack_root_error_component"})},[error]);return <div className="flex min-h-screen items-center justify-center bg-background"><button className="rounded-md bg-primary px-4 py-2 text-primary-foreground" onClick={()=>{router.invalidate();reset()}}>Retry system view</button></div>}
export const Route=createRootRouteWithContext<{queryClient:QueryClient}>()({head:()=>({meta:[{charSet:"utf-8"},{name:"viewport",content:"width=device-width, initial-scale=1"}],links:[{rel:"stylesheet",href:appCss},{rel:"icon",href:"/favicon.ico",type:"image/x-icon"},{rel:"preconnect",href:"https://fonts.googleapis.com"},{rel:"preconnect",href:"https://fonts.gstatic.com",crossOrigin:"anonymous"},{rel:"stylesheet",href:"https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Manrope:wght@400;500;600;700&display=swap"}]}),shellComponent:RootShell,component:RootComponent,notFoundComponent:NotFoundComponent,errorComponent:ErrorComponent});
function RootShell({children}:{children:ReactNode}){return <html lang="en"><head><HeadContent/></head><body>{children}<Scripts/></body></html>}
import { ThemeProvider } from "@/context/ThemeContext";
import { LanguageProvider } from "@/context/LanguageContext";

function RootComponent(){const {queryClient}=Route.useRouteContext();return <QueryClientProvider client={queryClient}><ThemeProvider><LanguageProvider><AppShell><Outlet/></AppShell></LanguageProvider></ThemeProvider></QueryClientProvider>}
