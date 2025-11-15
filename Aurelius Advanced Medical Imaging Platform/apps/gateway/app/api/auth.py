"""Authentication endpoints."""
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, EmailStr
from app.core.auth import keycloak_openid, get_current_user, User

router = APIRouter()


class LoginRequest(BaseModel):
    """Login request model."""
    username: str
    password: str


class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: str
    refresh_expires_in: int


class UserInfo(BaseModel):
    """User information model."""
    sub: str
    username: str
    email: str
    roles: list[str]
    given_name: str = ""
    family_name: str = ""


@router.post("/login", response_model=TokenResponse)
async def login(credentials: LoginRequest):
    """
    Authenticate user with Keycloak and return JWT token.
    
    Args:
        credentials: User credentials
        
    Returns:
        TokenResponse: Access and refresh tokens
        
    Raises:
        HTTPException: If authentication fails
    """
    try:
        # Get token from Keycloak
        token = keycloak_openid.token(
            credentials.username,
            credentials.password
        )
        
        return TokenResponse(
            access_token=token["access_token"],
            expires_in=token["expires_in"],
            refresh_token=token["refresh_token"],
            refresh_expires_in=token["refresh_expires_in"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(refresh_token: str):
    """
    Refresh access token using refresh token.
    
    Args:
        refresh_token: Refresh token
        
    Returns:
        TokenResponse: New access and refresh tokens
        
    Raises:
        HTTPException: If refresh fails
    """
    try:
        token = keycloak_openid.refresh_token(refresh_token)
        
        return TokenResponse(
            access_token=token["access_token"],
            expires_in=token["expires_in"],
            refresh_token=token["refresh_token"],
            refresh_expires_in=token["refresh_expires_in"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )


@router.post("/logout")
async def logout(
    refresh_token: str,
    user: User = Depends(get_current_user)
):
    """
    Logout user and invalidate tokens.
    
    Args:
        refresh_token: Refresh token to invalidate
        user: Current authenticated user
        
    Returns:
        Success message
    """
    try:
        keycloak_openid.logout(refresh_token)
        return {"message": "Successfully logged out"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Logout failed"
        )


@router.get("/me", response_model=UserInfo)
async def get_user_info(user: User = Depends(get_current_user)):
    """
    Get current user information.
    
    Args:
        user: Current authenticated user
        
    Returns:
        UserInfo: User information
    """
    return UserInfo(
        sub=user.sub,
        username=user.username,
        email=user.email,
        roles=user.roles,
        given_name=user.extra.get("given_name", ""),
        family_name=user.extra.get("family_name", "")
    )


@router.get("/verify")
async def verify_token(user: User = Depends(get_current_user)):
    """
    Verify JWT token validity.
    
    Args:
        user: Current authenticated user
        
    Returns:
        Verification status
    """
    return {
        "valid": True,
        "username": user.username,
        "roles": user.roles
    }
